"""P3 velocity definition, double-branch formula, gradients and isolated loading."""
import argparse
import gc
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from config import cfg
from model import make_model
from model.backbones.history_p3 import HistoryP3Adapter
from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from utils.history_training import history_gradient_groups, clip_history_gradients
from check_history import historical_module


def seed(value=1234):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def norm(parameters):
    return sum(float(p.grad.detach().abs().sum()) for p in parameters if p.grad is not None)


def formula_check():
    seed(1235)
    a = HistoryP3Adapter(16, 4, 4, hidden_dim=3)
    h = torch.randn(2, 5, 16)
    r1, hist, q, first = a.correction(h, 0, [], None, True, True)
    assert torch.count_nonzero(r1) == 0 and hist == [] and q is None and first['prediction'] is None
    assert a.velocity_gain.shape == (3, 1, 5, 16) and a.innovation_gain.shape == (2, 1, 4, 16)
    assert len(a.predictors) == 2 and not hasattr(a, 'coordinate') and not hasattr(a, 'anchors')
    assert all(torch.count_nonzero(p[-1].weight) == 0 and torch.count_nonzero(p[-1].bias) == 0 for p in a.predictors)
    with torch.no_grad():
        a.velocity_gain.fill_(.02); a.innovation_gain.fill_(.03)
        a.predictors[0][-1].bias.fill_(.13)
    v1 = torch.randn_like(h, requires_grad=True)
    r2, hist2, q2, item2 = a.correction(h, 1, hist, None, True, True, previous_velocity=v1)
    z2 = F.layer_norm(v1, (16,), eps=1e-5)
    assert torch.equal(r2, a.velocity_gain[0] * z2)
    assert item2['prediction'] is None and q2 is None and len(hist2) == 1
    assert torch.equal(hist2[0], z2[:, 1:].detach()) and not hist2[0].requires_grad
    v2 = torch.randn_like(h, requires_grad=True)
    r3, hist3, _, item3 = a.correction(h, 2, hist2, None, True, True, previous_velocity=v2)
    z3 = F.layer_norm(v2, (16,), eps=1e-5)
    prediction = a.predictors[0](torch.cat((hist2[-1], torch.zeros_like(hist2[-1])), -1))
    e = z3[:, 1:] - (hist2[-1] + prediction).detach()
    expected_v = a.velocity_gain[1] * z3
    expected_e = .25 * a.innovation_gain[0] * F.layer_norm(e, (16,), eps=1e-5)
    assert torch.equal(r3, expected_v + torch.cat((torch.zeros_like(expected_v[:, :1]), expected_e), 1))
    assert torch.equal(r3[:, :1], expected_v[:, :1]) and torch.count_nonzero(expected_e) > 0
    assert torch.equal(item3['target'], (z3[:, 1:] - z2[:, 1:]).detach())
    (r3 * torch.randn_like(r3)).sum().backward(retain_graph=True)
    assert v2.grad is not None and v1.grad is None and norm(a.predictors.parameters()) == 0
    assert norm([a.velocity_gain]) > 0 and norm([a.innovation_gain]) > 0
    a.zero_grad(set_to_none=True); v2.grad = None
    history_prediction_loss({'layers': {3: item3}, 'prediction_layers': [3]}).backward()
    assert norm(a.predictors.parameters()) > 0 and a.velocity_gain.grad is None and a.innovation_gain.grad is None
    assert v1.grad is None and v2.grad is None
    v3 = torch.randn_like(h)
    _, hist4, _, item4 = a.correction(h, 3, hist3, None, True, previous_velocity=v3)
    assert len(hist4) == 2
    assert torch.equal(item4['prediction'], a.predictors[1](torch.cat((hist3[-1], hist3[-1] - hist3[-2]), -1)))
    assert a.correction(h, 3, hist3, None, False, previous_velocity=v3)[3] is None
    c0 = DenseCrossLayerTokenTrajectory(4, 5, 16, 0.0)
    with torch.no_grad():
        c0.gain.copy_(a.velocity_gain); a.innovation_gain.zero_()
    assert torch.equal(a.correction(h, 2, hist2, None, previous_velocity=v2)[0], c0(2, v2, v1))
    p, q = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))
    p.grad, q.grad = torch.full_like(p, 100), torch.full_like(q, 200)
    clip_history_gradients({'task': [p], 'prediction': [q]})
    assert abs(float(p.grad.norm()) - 1) < 1e-5 and abs(float(q.grad.norm()) - 1) < 1e-5


def check_actual_velocity(model, images):
    adapter = model.base.history_adapter
    inputs, outputs, before_writes = {}, {}, {}
    handles = []
    for index, block in enumerate(model.base.blocks):
        def before(module, args, i=index): inputs[i] = args[0].detach().clone()
        def after(module, args, result, i=index): outputs[i] = result.detach().clone()
        handles.extend([block.register_forward_pre_hook(before), block.register_forward_hook(after)])
    original = adapter.correction
    confirmed = []
    def capture(h, layer, history, previous_r, *args, **kwargs):
        before_writes[layer] = h.detach().clone()
        if layer > 0:
            velocity = kwargs['previous_velocity']
            assert torch.equal(velocity, outputs[layer - 1] - inputs[layer - 1])
            if layer >= 2:
                assert not torch.equal(velocity, outputs[layer - 1] - before_writes[layer - 1])
            confirmed.append(layer)
        return original(h, layer, history, previous_r, *args, **kwargs)
    adapter.correction = capture
    model.eval()
    with torch.no_grad():
        adapter.velocity_gain.fill_(.002); adapter.innovation_gain.fill_(.003)
        model(images[0], mode=1)
        adapter.velocity_gain.zero_(); adapter.innovation_gain.zero_()
    adapter.correction = original
    for handle in handles: handle.remove()
    assert confirmed == list(range(1, 12))


def legacy_check(args):
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_HI1.yml'))
    c.HISTORY.METHOD = 'legacy'
    c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
    if args.pretrained: c.MODEL.PRETRAIN_PATH = args.pretrained
    old_vit = historical_module('model/backbones/vit_pytorch.py', 'model.backbones.p3_legacy_vit', '30eaccc')
    old = historical_module('model/make_model.py', 'model.p3_legacy_model', '30eaccc')
    old.__dict__['__factory_T_type']['vit_base_clip'] = old_vit.vit_base_clip
    seed(); reference = old.make_model(c, 500, 7, 2)
    seed(); current = make_model(c, 500, 7, 2)
    assert current.base.history_adapter.coordinate.shape == (768, 64)
    assert reference.state_dict().keys() == current.state_dict().keys()
    assert all(torch.equal(v, current.state_dict()[k]) for k, v in reference.state_dict().items())
    with torch.no_grad():
        reference.base.history_adapter.gain.fill_(.002); current.base.history_adapter.gain.fill_(.002)
        images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
        seed(); left = reference(images)
        seed(); right = current(images)
    assert all(torch.equal(x, y) for x, y in zip(left[:3], right[:3]))
    assert torch.equal(history_prediction_loss(left[3]), history_prediction_loss(right[3]))
    old_solver = historical_module('solver/make_optimizer.py', 'solver.p3_legacy_optimizer', '30eaccc')
    old_opt, _ = old_solver.make_optimizer(c, reference, nn.Linear(1, 1))
    new_opt, _ = make_optimizer(c, current, nn.Linear(1, 1))
    assert [(g['lr'], g['weight_decay']) for g in old_opt.param_groups] == [(g['lr'], g['weight_decay']) for g in new_opt.param_groups]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / 'legacy_report.json').write_text(json.dumps({'legacy_hi1_unchanged': True, 'rank': 64, 'nonzero_gain_forward_and_loss_equal': True, 'optimizer_equal': True}, indent=2), encoding='utf-8')
    print('P3_LEGACY_CHECK_OK')
    return True


def run(args):
    torch.set_num_threads(4)
    if args.legacy_only:
        legacy_check(args)
        return
    formula_check()
    report = {'status': 'LOCAL_CHECK_ONLY', 'torch': torch.__version__, 'source_base': '30eaccc',
              'formula_layer_boundaries_and_direct_gradients': True, 'separate_clipping': True, 'formal_training': False}
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_P3.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
    if args.pretrained: c.MODEL.PRETRAIN_PATH = args.pretrained
    bc = c.clone(); bc.HISTORY.ENABLED = False
    old_vit = historical_module('model/backbones/vit_pytorch.py', 'model.backbones.p3_reference_vit', '30eaccc')
    old = historical_module('model/make_model.py', 'model.p3_reference_model', '30eaccc')
    old.__dict__['__factory_T_type']['vit_base_clip'] = old_vit.vit_base_clip
    old_solver = historical_module('solver/make_optimizer.py', 'solver.p3_reference_optimizer', '30eaccc')
    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    for mode in ('baseline', 'c0'):
        rc = bc.clone(); rc.MODEL.TOKEN_TRAJECTORY = mode == 'c0'; rc.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX = 0.0
        seed(); reference = old.make_model(rc, 500, 7, 2)
        seed(); current = make_model(rc, 500, 7, 2)
        assert reference.state_dict().keys() == current.state_dict().keys()
        assert all(torch.equal(v, current.state_dict()[k]) for k, v in reference.state_dict().items())
        if mode == 'c0':
            with torch.no_grad():
                reference.base.token_trajectory.gain.fill_(.002); current.base.token_trajectory.gain.fill_(.002)
        with torch.no_grad():
            seed(); left = reference(images)
            seed(); right = current(images)
        assert all(torch.equal(x, y) for x, y in zip(left, right))
        old_opt, _ = old_solver.make_optimizer(rc, reference, nn.Linear(1, 1))
        new_opt, _ = make_optimizer(rc, current, nn.Linear(1, 1))
        assert [(g['lr'], g['weight_decay']) for g in old_opt.param_groups] == [(g['lr'], g['weight_decay']) for g in new_opt.param_groups]
        report[mode + '_unchanged'] = True
        del reference, current, old_opt, new_opt, left, right; gc.collect()
    seed(); baseline = make_model(bc, 500, 7, 2)
    rng = torch.random.get_rng_state().clone()
    seed(); model = make_model(c, 500, 7, 2)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
    with torch.no_grad():
        seed(); left = baseline(images)
        seed(); right = model(images, history_diagnostics=True)
    assert all(torch.equal(x, y) for x, y in zip(left, right[:3]))
    assert right[3]['prediction_layers'] == list(range(3, 13))
    assert right[3]['layers'][1]['prediction'] is None and right[3]['layers'][2]['prediction'] is None
    assert all(right[3]['layers'][l]['target'].shape == (6, 128, 768) for l in range(3, 13))
    mean_loss = torch.stack([F.smooth_l1_loss(right[3]['layers'][l]['prediction'], right[3]['layers'][l]['target']) for l in range(3, 13)]).mean()
    assert torch.equal(mean_loss, history_prediction_loss(right[3]))
    report['zero_gain_baseline_rng_and_ten_layer_loss'] = True
    del baseline, left, right; gc.collect()
    # Nonzero velocity branch must reproduce C0's complete forward when innovation gain is zero.
    rc = bc.clone(); rc.MODEL.TOKEN_TRAJECTORY = True; rc.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX = 0.0
    seed(); c0 = old.make_model(rc, 500, 7, 2)
    with torch.no_grad():
        c0.base.token_trajectory.gain.fill_(.002); model.base.history_adapter.velocity_gain.fill_(.002)
        seed(); left = c0(images)
        seed(); right = model(images)
    assert all(torch.equal(x, y) for x, y in zip(left, right[:3]))
    report['nonzero_velocity_only_equals_c0_full_forward'] = True
    del c0, left, right; gc.collect()
    check_actual_velocity(model, images)
    report['actual_block_output_minus_corrected_input'] = True
    adapter = model.base.history_adapter
    assert sum(p.numel() for p in adapter.parameters()) == 2818112 and model.base.token_trajectory is None
    if args.real_data:
        from datasets import make_dataloader
        dc = c.clone(); dc.DATASETS.ROOT_DIR = args.real_data
        dc.DATALOADER.NUM_WORKERS = 0; dc.SOLVER.IMS_PER_BATCH = 4; dc.DATALOADER.NUM_INSTANCE = 2
        seed(); loader, _, _, _, classes, _, _ = make_dataloader(dc)
        images, labels, _ = next(iter(loader)); assert classes == 500
        report['real_batch'] = {'actual_images': sum(len(x) for x in images), 'identities': labels.unique().numel()}
    else: labels = torch.tensor([0, 1])
    target = labels.repeat(3)
    optimizer, _ = make_optimizer(c, model, nn.Linear(1, 1))
    by_id = {id(p): g for g in optimizer.param_groups for p in g['params']}
    for name, p in adapter.named_parameters():
        family = name.split('.')[0]; group = by_id[id(p)]
        assert group['lr'] == (1e-4 if family == 'predictors' else 3.5e-4)
        assert group['weight_decay'] == (0 if family == 'innovation_gain' else 1e-4)
    groups = history_gradient_groups(model)
    assert not ({id(p) for p in groups['task']} & {id(p) for p in groups['prediction']})
    assert sum(len(v) for v in groups.values()) == sum(p.requires_grad for p in model.parameters())
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        seed(); out = model(images, history_diagnostics=True)
        task = F.cross_entropy(out[0], target) + TripletLoss()(out[1], target)[0]
        prediction = history_prediction_loss(out[3])
        task.backward(retain_graph=True)
        assert norm(adapter.predictors.parameters()) == 0 and norm([adapter.velocity_gain]) > 0 and norm([adapter.innovation_gain]) > 0
        assert model.base.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
        prediction.backward()
        assert norm(adapter.predictors.parameters()) > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clip_history_gradients(groups); optimizer.step()
    report['gain_startup_lr_wd_and_real_updates'] = True
    report['adapter_parameters'] = sum(p.numel() for p in adapter.parameters())
    model.zero_grad(set_to_none=True)
    del optimizer, out, task, prediction, groups; gc.collect()
    model.eval()
    with torch.no_grad():
        first = model(images[0], mode=1); model(images[1], mode=2)
        assert torch.equal(first, model(images[0], mode=1))
        model.neck_feat = 'before'; pre = model(images[0], mode=1)
        assert torch.equal(pre, model.base(images[0])); model.neck_feat = 'after'
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.pth'; torch.save(model.state_dict(), checkpoint)
    with torch.no_grad(): adapter.innovation_gain.add_(1)
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    model.load_param(str(checkpoint))
    with torch.no_grad(): assert torch.equal(first, model(images[0], mode=1))
    assert not torch.equal(pre, first)
    spec = adapter.specification.clone()
    with torch.no_grad(): adapter.specification[-1] = 1.
    try: model.load_param(str(checkpoint))
    except ValueError: report['wrong_innovation_alpha_rejected'] = True
    else: raise AssertionError('P3 alpha specification was not checked')
    adapter.specification.copy_(spec)
    report['strict_reload_readouts_and_state_reset'] = True
    report['legacy_hi1_unchanged'] = legacy_check(args)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('P3_CHECK_OK ' + json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', default='')
    p.add_argument('--real-data', default='')
    p.add_argument('--output', default=str(ROOT / 'work/p3_check'))
    p.add_argument('--legacy-only', action='store_true')
    run(p.parse_args())
