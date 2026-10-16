"""P3-C formula, metadata routing, original-P3 limit and real-data updates."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from config import cfg
from model import make_model
from model.backbones.history_p3_c import HistoryP3CAdapter
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from utils.history_training import history_gradient_groups, clip_history_gradients
from check_history import historical_module
from check_history_p3 import seed, norm, formula_check


def condition_formula():
    seed(1235)
    a = HistoryP3CAdapter(16, 4, 4, 3)
    assert a.modal_condition.shape == (2, 3, 3) and a.platform_condition.shape == (2, 2, 3)
    assert torch.count_nonzero(a.modal_condition) == torch.count_nonzero(a.platform_condition) == 0
    conditions = (torch.tensor([0, 2]), torch.tensor([0, 1]))
    last, delta = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    target = torch.randn_like(last)
    loss = F.smooth_l1_loss(a._predict_history(0, last, delta, conditions), target)
    loss.backward()
    assert norm([a.modal_condition, a.platform_condition]) == 0  # W2 starts at zero.
    assert norm(a.predictors.parameters()) > 0
    a.zero_grad(set_to_none=True)
    with torch.no_grad():
        a.modal_condition.normal_(std=.2); a.platform_condition.normal_(std=.3)
        a.predictors[0][-1].weight.normal_(std=.02)
        a.velocity_gain.fill_(.02); a.innovation_gain.fill_(.03)
    expected = F.linear(F.gelu(F.linear(torch.cat((last, delta), -1),
                        a.predictors[0][0].weight, a.predictors[0][0].bias)
                        + a.modal_condition[0, conditions[0]][:, None]
                        + a.platform_condition[0, conditions[1]][:, None]),
                        a.predictors[0][2].weight, a.predictors[0][2].bias)
    assert torch.equal(expected, a._predict_history(0, last, delta, conditions))
    h, velocity = torch.randn(2, 5, 16), torch.randn(2, 5, 16, requires_grad=True)
    r, hist, _, item = a.correction(h, 2, [last], None, True, True,
                                    previous_velocity=velocity, conditions=conditions)
    z = F.layer_norm(velocity, (16,), eps=1e-5)
    prediction = a._predict_history(0, last, torch.zeros_like(last), conditions)
    rv = a.velocity_gain[1] * z
    re = .25 * a.innovation_gain[0] * F.layer_norm(z[:, 1:] - (last + prediction).detach(), (16,), eps=1e-5)
    assert torch.equal(r, rv + torch.cat((torch.zeros_like(rv[:, :1]), re), 1))
    assert torch.equal(r[:, :1], rv[:, :1]) and all(not t.requires_grad for t in hist)
    (r * torch.randn_like(r)).sum().backward()
    assert norm(a.predictors.parameters()) == norm([a.modal_condition, a.platform_condition]) == 0
    assert velocity.grad is not None
    a.zero_grad(set_to_none=True); velocity.grad = None
    history_prediction_loss({'layers': {3: item}, 'prediction_layers': [3]}).backward()
    assert norm(a.predictors.parameters()) > 0
    assert norm([a.modal_condition]) > 0 and norm([a.platform_condition]) > 0
    assert velocity.grad is None and a.velocity_gain.grad is None and a.innovation_gain.grad is None
    assert torch.count_nonzero(a.modal_condition.grad[1]) == 0
    assert torch.count_nonzero(a.modal_condition.grad[0, 1]) == 0


def p3_limit(c, images, cameras, report):
    old_model = historical_module('model/make_model.py', 'model.p3c_reference_model', '2c71241')
    old_adapter = historical_module('model/backbones/history_p3.py', 'model.backbones.p3c_reference_adapter', '2c71241')
    pc = c.clone(); pc.HISTORY.METHOD = 'p3'
    seed(); old = old_model.make_model(pc, 500, 7, 2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1235)
        old.base.history_adapter = old_adapter.HistoryP3Adapter(768, 12, 128, 32, .25, range(3, 13))
    rng = torch.random.get_rng_state().clone()
    seed(); current = make_model(c, 500, 7, 2)
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, value in old.state_dict().items():
        if k != 'base.history_adapter.specification':
            assert torch.equal(value, current.state_dict()[k]), k
    extra = set(dict(current.named_parameters())) - set(dict(old.named_parameters()))
    assert extra == {'base.history_adapter.modal_condition', 'base.history_adapter.platform_condition'}
    # Nonzero gains and W2 avoid a vacuous zero-write/zero-predictor comparison.
    with torch.no_grad():
        old.base.history_adapter.velocity_gain.fill_(.002)
        old.base.history_adapter.innovation_gain.fill_(.003)
        for predictor in old.base.history_adapter.predictors:
            predictor[-1].weight.normal_(std=.002); predictor[-1].bias.normal_(std=.01)
        for key, value in old.state_dict().items():
            if key != 'base.history_adapter.specification': current.state_dict()[key].copy_(value)
        seed(); left = old(images, history_diagnostics=True)
        seed(); right = current(images, camids=cameras, history_diagnostics=True)
        assert all(torch.equal(x, y) for x, y in zip(left[:3], right[:3]))
        assert torch.equal(history_prediction_loss(left[3]), history_prediction_loss(right[3]))
        assert all(torch.equal(left[3]['layers'][i]['prediction'], right[3]['layers'][i]['prediction']) for i in range(3, 13))
    report['zero_conditions_equal_original_p3_nonzero_forward_loss_rng'] = True
    del old, current, left, right; gc.collect()


def routing_and_velocity(model, images, cameras, val_loaders, num_querys, report):
    adapter = model.base.history_adapter
    modality, platform = model._history_conditions(images, cameras, 0)
    assert torch.equal(modality, torch.arange(3).repeat_interleave(len(images[0])))
    cams = torch.cat(cameras)
    assert torch.equal(platform, (~((cams == 5) | (cams == 6))).long())
    for missing in (None, torch.tensor([-1, 0]), torch.tensor([7, 0]), torch.tensor([5., 0.])):
        try: model._history_conditions(images[0][:2], missing, 1)
        except ValueError: pass
        else: raise AssertionError('missing/invalid camera labels accepted')
    try: model._history_conditions(images[0], cameras[0], 4)
    except ValueError: pass
    else: raise AssertionError('invalid modality accepted')
    with torch.no_grad():
        adapter.velocity_gain.fill_(.002); adapter.innovation_gain.fill_(.003)
        adapter.modal_condition.normal_(std=.2); adapter.platform_condition.normal_(std=.3)
        for predictor in adapter.predictors: predictor[-1].weight.normal_(std=.002)
    model.eval()
    # Read one actual query and one actual gallery image from EACH modality.
    from datasets.make_dataloader import val_collate_fn
    own_predict = adapter._predict_history
    seen = []
    def capture(index, last, delta, conditions=None):
        seen.append((index, conditions[0].clone(), conditions[1].clone()))
        return own_predict(index, last, delta, conditions)
    adapter._predict_history = capture
    for mode, (loader, nquery) in enumerate(zip(val_loaders, num_querys), 1):
        batch = val_collate_fn([loader.dataset[0], loader.dataset[nquery]])
        image, camera = batch[0], batch[3]
        seen.clear()
        with torch.no_grad(): joint = model(image, camids=camera, mode=mode)
        expected_platform = (~((camera == 5) | (camera == 6))).long()
        assert len(seen) == 10 and all(torch.equal(m, torch.full_like(m, mode - 1)) and torch.equal(p, expected_platform) for _, m, p in seen)
        with torch.no_grad():
            single = torch.cat([model(image[i:i+1], camids=camera[i:i+1], mode=mode) for i in range(2)])
            reverse = model(image.flip(0), camids=camera.flip(0), mode=mode).flip(0)
        assert torch.allclose(joint, single, atol=1e-4, rtol=1e-4)
        assert torch.allclose(joint, reverse, atol=1e-5, rtol=1e-5)
    adapter._predict_history = own_predict
    # Same image, different own metadata: both conditions must affect inference.
    with torch.no_grad():
        probe = images[0][:1]
        rgb_aerial = model(probe, camids=torch.tensor([5]), mode=1)
        rgb_ground = model(probe, camids=torch.tensor([0]), mode=1)
        ir_aerial = model(probe, camids=torch.tensor([5]), mode=2)
        assert not torch.equal(rgb_aerial, rgb_ground) and not torch.equal(rgb_aerial, ir_aerial)
    # Check velocity against actual corrected block inputs, not raw neighboring states.
    inputs, outputs, before_writes, handles, confirmed = {}, {}, {}, [], []
    for i, block in enumerate(model.base.blocks):
        def before(module, args, index=i): inputs[index] = args[0].detach().clone()
        def after(module, args, result, index=i): outputs[index] = result.detach().clone()
        handles.extend([block.register_forward_pre_hook(before), block.register_forward_hook(after)])
    original = adapter.correction
    def correction(h, layer, history, previous_r, *args, **kwargs):
        before_writes[layer] = h.detach().clone()
        if layer > 0:
            assert torch.equal(kwargs['previous_velocity'], outputs[layer - 1] - inputs[layer - 1])
            if layer >= 2: assert not torch.equal(kwargs['previous_velocity'], outputs[layer - 1] - before_writes[layer - 1])
            confirmed.append(layer)
        return original(h, layer, history, previous_r, *args, **kwargs)
    adapter.correction = correction
    with torch.no_grad(): model(images[0], camids=cameras[0], mode=1)
    adapter.correction = original
    for handle in handles: handle.remove()
    assert confirmed == list(range(1, 12))
    report['own_metadata_for_real_query_and_gallery_all_modalities'] = True
    report['batch_split_permutation_and_condition_effect'] = True
    report['invalid_missing_metadata_rejected_and_actual_velocity_verified'] = True


def main(args):
    torch.set_num_threads(4)
    formula_check(); condition_formula()
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_P3_C.yml'))
    c.MODEL.PRETRAIN_PATH = args.pretrained
    c.DATASETS.ROOT_DIR = args.real_data; c.DATALOADER.NUM_WORKERS = 0
    c.SOLVER.IMS_PER_BATCH = 4; c.DATALOADER.NUM_INSTANCE = 2
    from datasets import make_dataloader
    seed(); loader, _, vals, nqueries, classes, cameras_n, views_n = make_dataloader(c)
    images, labels, cameras = next(iter(loader))
    assert (classes, cameras_n) == (500, 7)
    report = {'status': 'LOCAL_CPU_CHECK_ONLY', 'source_base': '2c71241', 'formal_training': False,
              'condition_formula_and_delayed_startup_gradients': True, 'actual_images': sum(len(x) for x in images)}
    p3_limit(c, images, cameras, report)
    seed(); model = make_model(c, classes, cameras_n, views_n)
    adapter = model.base.history_adapter
    assert sum(p.numel() for p in adapter.parameters()) == 2819712
    optimizer, _ = make_optimizer(c, model, nn.Linear(1, 1))
    by_id = {id(p): g for g in optimizer.param_groups for p in g['params']}
    for name, p in adapter.named_parameters():
        family = name.split('.')[0]; group = by_id[id(p)]
        assert group['lr'] == (1e-4 if family in ('predictors', 'modal_condition', 'platform_condition') else 3.5e-4)
        assert group['weight_decay'] == (0 if family in ('innovation_gain', 'modal_condition', 'platform_condition') else 1e-4)
    groups = history_gradient_groups(model)
    assert sum(p.numel() for p in groups['prediction']) == 746880
    task_ids = {id(p) for p in groups['task']}; pred_ids = {id(p) for p in groups['prediction']}
    assert not task_ids & pred_ids and task_ids | pred_ids == {id(p) for p in model.parameters() if p.requires_grad}
    target = labels.repeat(3)
    for step in range(2):
        model.train(); optimizer.zero_grad(set_to_none=True)
        seed(); out = model(images, camids=cameras, history_diagnostics=True)
        task = F.cross_entropy(out[0], target) + TripletLoss()(out[1], target)[0]
        prediction = history_prediction_loss(out[3])
        manual = torch.stack([F.smooth_l1_loss(out[3]['layers'][i]['prediction'], out[3]['layers'][i]['target']) for i in range(3, 13)]).mean()
        assert torch.equal(prediction, manual)
        task.backward(retain_graph=True)
        assert norm(groups['prediction']) == 0
        assert norm([adapter.velocity_gain]) > 0 and norm([adapter.innovation_gain]) > 0
        task_before = [None if p.grad is None else p.grad.clone() for p in groups['task']]
        prediction.backward()
        assert all((p.grad is None if g is None else torch.equal(p.grad, g)) for p, g in zip(groups['task'], task_before))
        assert norm(adapter.predictors.parameters()) > 0
        for p in (adapter.modal_condition, adapter.platform_condition):
            assert (norm([p]) == 0) if step == 0 else (norm([p]) > 0)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clip_history_gradients(groups); optimizer.step()
    assert torch.count_nonzero(adapter.modal_condition) > 0 and torch.count_nonzero(adapter.platform_condition) > 0
    report.update(adapter_parameters=2819712, added_condition_parameters=1600, prediction_parameters=746880,
                  real_cpu_optimizer_updates=2, gradient_isolation_lr_wd_and_loss_reduction=True)
    del optimizer, out, task, prediction, manual, task_before, groups; model.zero_grad(set_to_none=True); gc.collect()
    routing_and_velocity(model, images, cameras, vals, nqueries, report)
    model.eval(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        first = model(images[0], camids=cameras[0], mode=1)
        model(images[1], camids=cameras[1], mode=2)
        assert torch.equal(first, model(images[0], camids=cameras[0], mode=1))
        model.neck_feat = 'before'; pre = model(images[0], camids=cameras[0], mode=1)
        direct = model.base(images[0], history_conditions=model._history_conditions(images[0], cameras[0], 1))
        assert torch.equal(pre, direct); model.neck_feat = 'after'
    checkpoint = output / 'checkpoint.pth'; torch.save(model.state_dict(), checkpoint)
    with torch.no_grad(): adapter.modal_condition.add_(1)
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    model.load_param(str(checkpoint))
    with torch.no_grad(): assert torch.equal(first, model(images[0], camids=cameras[0], mode=1))
    assert not torch.equal(first, pre)
    spec = adapter.specification.clone()
    with torch.no_grad(): adapter.specification[-1] = 0
    try: model.load_param(str(checkpoint))
    except ValueError: pass
    else: raise AssertionError('condition mapping specification mismatch accepted')
    adapter.specification.copy_(spec)
    report['strict_reload_readouts_and_forward_state_reset'] = True
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('P3_C_CHECK_OK ' + json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', required=True)
    p.add_argument('--real-data', required=True)
    p.add_argument('--output', default=str(ROOT / 'work/p3_c_check'))
    main(p.parse_args())
