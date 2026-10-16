"""P3-A formula, startup, gradient ownership, P3 parity and checkpoint checks."""
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
from model.backbones.history_p3_a import HistoryP3AAdapter
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from utils.history_training import history_gradient_groups, clip_history_gradients
from check_history import historical_module
from check_history_p3 import seed, norm, formula_check, check_actual_velocity, legacy_check


def retention_formula_check():
    seed(1235)
    a = HistoryP3AAdapter(16, 4, 4, hidden_dim=3)
    assert a.history_retention.shape == (2, 1, 1, 16)
    assert torch.equal(a.history_retention, torch.ones_like(a.history_retention))
    h, v1 = torch.randn(2, 5, 16), torch.randn(2, 5, 16, requires_grad=True)
    r1, hist, _, _ = a.correction(h, 0, [], None, True)
    assert not hist and torch.count_nonzero(r1) == 0
    _, hist2, _, item2 = a.correction(h, 1, hist, None, True, previous_velocity=v1)
    assert item2['prediction'] is None and len(hist2) == 1 and not hist2[0].requires_grad
    v2 = torch.randn_like(v1, requires_grad=True)
    # With zero Predictor output and zero gains, a already receives prediction data gradients.
    _, _, _, startup = a.correction(h, 2, hist2, None, True, previous_velocity=v2)
    history_prediction_loss({'layers': {3: startup}, 'prediction_layers': [3]}).backward()
    assert norm([a.history_retention]) > 0 and norm([a.velocity_gain, a.innovation_gain]) == 0
    assert v1.grad is None and v2.grad is None
    a.zero_grad(set_to_none=True)
    with torch.no_grad():
        a.velocity_gain.fill_(.02); a.innovation_gain.fill_(.03)
        a.history_retention.copy_(torch.linspace(-.5, 1.5, 16).repeat(2, 1, 1, 1))
        a.predictors[0][-1].weight.normal_(std=.02)
        a.predictors[0][-1].bias.normal_(std=.03)
    r3, hist3, _, item3 = a.correction(h, 2, hist2, None, True, True, previous_velocity=v2)
    last = hist2[-1]
    nonlinear = a.predictors[0](torch.cat((last, torch.zeros_like(last)), -1))
    predicted_state = a.history_retention[0] * last + nonlinear
    full_delta = (a.history_retention[0] - 1) * last + nonlinear
    z3 = F.layer_norm(v2, (16,), eps=1e-5)
    innovation = z3[:, 1:] - predicted_state.detach()
    rv = a.velocity_gain[1] * z3
    re = .25 * a.innovation_gain[0] * F.layer_norm(innovation, (16,), eps=1e-5)
    assert torch.equal(item3['prediction'], full_delta)
    assert torch.equal(item3['target'], (z3[:, 1:] - last).detach())
    assert torch.equal(r3, rv + torch.cat((torch.zeros_like(rv[:, :1]), re), 1))
    assert torch.equal(r3[:, :1], rv[:, :1])
    assert not torch.equal(full_delta, nonlinear)  # The retention term is actually supervised.
    (r3 * torch.randn_like(r3)).sum().backward()
    assert norm([a.history_retention]) == norm(a.predictors.parameters()) == 0
    assert norm([a.velocity_gain]) > 0 and norm([a.innovation_gain]) > 0
    assert v2.grad is not None and v1.grad is None
    a.zero_grad(set_to_none=True); v2.grad = None
    history_prediction_loss({'layers': {3: item3}, 'prediction_layers': [3]}).backward()
    assert norm([a.history_retention]) > 0 and norm(a.predictors.parameters()) > 0
    assert a.velocity_gain.grad is None and a.innovation_gain.grad is None
    assert v1.grad is None and v2.grad is None
    v3 = torch.randn_like(h)
    _, hist4, _, item4 = a.correction(h, 3, hist3, None, True, previous_velocity=v3)
    mlp4 = a.predictors[1](torch.cat((hist3[-1], hist3[-1] - hist3[-2]), -1))
    assert torch.equal(item4['prediction'], (a.history_retention[1] - 1) * hist3[-1] + mlp4)
    assert len(hist4) == 2 and all(not x.requires_grad for x in hist4)
    assert a.correction(h, 3, hist3, None, False, previous_velocity=v3)[3] is None


def full_p3_parity(c, images, report):
    pc = c.clone(); pc.HISTORY.METHOD = 'p3'
    old = historical_module('model/make_model.py', 'model.p3a_reference_model', '2c71241')
    old_adapter = historical_module('model/backbones/history_p3.py', 'model.backbones.p3a_reference_adapter', '2c71241')
    seed(); reference = old.make_model(pc, 500, 7, 2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(pc.HISTORY.INIT_SEED)
        reference.base.history_adapter = old_adapter.HistoryP3Adapter(768, 12, 128, 32, .25, list(range(3, 13)))
    reference_rng = torch.random.get_rng_state().clone()
    seed(); candidate = make_model(c, 500, 7, 2)
    assert torch.equal(reference_rng, torch.random.get_rng_state())
    for key,value in reference.state_dict().items():
        if key != 'base.history_adapter.specification':
            assert torch.equal(value, candidate.state_dict()[key]), key
    # Also compare a nonzero, nontrivial Predictor and both residual branches.
    with torch.no_grad():
        reference.base.history_adapter.velocity_gain.fill_(.002)
        reference.base.history_adapter.innovation_gain.fill_(.003)
        for predictor in reference.base.history_adapter.predictors:
            predictor[-1].weight.normal_(std=.002)
            predictor[-1].bias.normal_(std=.01)
        for key,value in reference.state_dict().items():
            if key != 'base.history_adapter.specification': candidate.state_dict()[key].copy_(value)
        seed(); left = reference(images, history_diagnostics=True)
        seed(); right = candidate(images, history_diagnostics=True)
    assert all(torch.equal(x,y) for x,y in zip(left[:3],right[:3]))
    assert torch.equal(history_prediction_loss(left[3]), history_prediction_loss(right[3]))
    # New P3 dispatch/helper must preserve the original P3 source and optimization rules.
    seed(); current_p3 = make_model(pc, 500, 7, 2)
    current_p3.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        seed(); current_out = current_p3(images, history_diagnostics=True)
    assert all(torch.equal(x,y) for x,y in zip(left[:3],current_out[:3]))
    assert torch.equal(history_prediction_loss(left[3]), history_prediction_loss(current_out[3]))
    old_solver = historical_module('solver/make_optimizer.py', 'solver.p3a_reference_optimizer', '2c71241')
    old_opt,_ = old_solver.make_optimizer(pc, reference, nn.Linear(1,1))
    current_opt,_ = make_optimizer(pc, current_p3, nn.Linear(1,1))
    assert [(g['lr'],g['weight_decay']) for g in old_opt.param_groups] == [(g['lr'],g['weight_decay']) for g in current_opt.param_groups]
    report['original_p3_source_forward_loss_optimizer_unchanged'] = True
    report['a_one_nonzero_full_forward_and_loss_equal_p3'] = True
    report['shared_initial_parameters_and_rng_equal_p3'] = True
    del reference,candidate,current_p3,old_opt,current_opt,left,right,current_out
    gc.collect()


def run(args):
    torch.set_num_threads(4)
    formula_check()  # Regression of the unchanged P3/C0 local formula and clipping.
    retention_formula_check()
    report = {'status':'LOCAL_CHECK_ONLY','torch':torch.__version__,'source_base':'2c71241',
              'formula_full_increment_and_startup_gradient':True,'formal_training':False}
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_P3_A.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
    if args.pretrained: c.MODEL.PRETRAIN_PATH = args.pretrained
    images = [torch.randn(2,3,256,128) for _ in range(3)]
    full_p3_parity(c,images,report)
    bc = c.clone(); bc.HISTORY.ENABLED = False
    seed(); baseline = make_model(bc,500,7,2)
    rng = torch.random.get_rng_state().clone()
    seed(); model = make_model(c,500,7,2)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
    with torch.no_grad():
        seed(); left = baseline(images)
        seed(); right = model(images,history_diagnostics=True)
    assert all(torch.equal(x,y) for x,y in zip(left,right[:3]))
    assert right[3]['prediction_layers'] == list(range(3,13))
    assert all(right[3]['layers'][l]['prediction'] is None for l in (1,2))
    assert all(right[3]['layers'][l]['target'].shape == (6,128,768) for l in range(3,13))
    expected = torch.stack([F.smooth_l1_loss(right[3]['layers'][l]['prediction'],right[3]['layers'][l]['target']) for l in range(3,13)]).mean()
    assert torch.equal(expected,history_prediction_loss(right[3]))
    report['zero_gain_baseline_rng_and_ten_layer_loss'] = True
    del baseline,left,right; gc.collect()
    check_actual_velocity(model,images)
    report['actual_block_output_minus_corrected_input'] = True
    adapter = model.base.history_adapter
    assert sum(p.numel() for p in adapter.parameters()) == 2825792
    if args.real_data:
        from datasets import make_dataloader
        dc = c.clone(); dc.DATASETS.ROOT_DIR = args.real_data
        dc.DATALOADER.NUM_WORKERS = 0; dc.SOLVER.IMS_PER_BATCH = 4; dc.DATALOADER.NUM_INSTANCE = 2
        seed(); loader,_,_,_,classes,_,_ = make_dataloader(dc)
        images,labels,_ = next(iter(loader)); assert classes == 500
        report['real_batch'] = {'actual_images':sum(len(x) for x in images),'identities':labels.unique().numel()}
    else: labels = torch.tensor([0,1])
    target = labels.repeat(3)
    optimizer,_ = make_optimizer(c,model,nn.Linear(1,1))
    by_id = {id(p):g for g in optimizer.param_groups for p in g['params']}
    for name,p in adapter.named_parameters():
        family = name.split('.')[0]; g = by_id[id(p)]
        assert g['lr'] == (1e-4 if family in ('predictors','history_retention') else 3.5e-4)
        assert g['weight_decay'] == (0 if family in ('innovation_gain','history_retention') else 1e-4)
    groups = history_gradient_groups(model)
    assert not ({id(p) for p in groups['task']} & {id(p) for p in groups['prediction']})
    assert sum(len(v) for v in groups.values()) == sum(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in groups['prediction']) == 752960
    assert id(adapter.history_retention) in {id(p) for p in groups['prediction']}
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        seed(); out = model(images,history_diagnostics=True)
        task = F.cross_entropy(out[0],target) + TripletLoss()(out[1],target)[0]
        prediction = history_prediction_loss(out[3])
        task.backward(retain_graph=True)
        assert norm(adapter.predictors.parameters()) == norm([adapter.history_retention]) == 0
        assert norm([adapter.velocity_gain]) > 0 and norm([adapter.innovation_gain]) > 0
        task_grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        prediction.backward()
        assert norm(adapter.predictors.parameters()) > 0 and norm([adapter.history_retention]) > 0
        assert all(torch.equal(task_grads[n],p.grad) for n,p in model.named_parameters() if n in task_grads)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clip_history_gradients(groups); optimizer.step()
    assert torch.any(adapter.history_retention != 1)
    report['prediction_only_retention_lr_wd_separate_clip_and_real_updates'] = True
    report['adapter_parameters'] = sum(p.numel() for p in adapter.parameters())
    model.zero_grad(set_to_none=True)
    del optimizer,out,task,prediction,groups,task_grads; gc.collect()
    model.eval()
    with torch.no_grad():
        first = model(images[0],mode=1); model(images[1],mode=2)
        assert torch.equal(first,model(images[0],mode=1))
        model.neck_feat = 'before'; pre = model(images[0],mode=1)
        assert torch.equal(pre,model.base(images[0])); model.neck_feat = 'after'
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    checkpoint = output/'checkpoint.pth'; torch.save(model.state_dict(),checkpoint)
    with torch.no_grad(): adapter.history_retention.add_(1)
    model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'),strict=True)
    model.load_param(str(checkpoint))
    with torch.no_grad(): assert torch.equal(first,model(images[0],mode=1))
    assert not torch.equal(first,pre)
    bad = dict(torch.load(str(checkpoint),map_location='cpu'))
    bad.pop('base.history_adapter.history_retention')
    wrong = output/'missing_retention.pth'; torch.save(bad,wrong)
    try: model.load_param(str(wrong))
    except ValueError: report['missing_retention_rejected'] = True
    else: raise AssertionError('Missing retention tensor was silently accepted')
    spec = adapter.specification.clone()
    adapter.specification[0] = 103
    try: model.load_param(str(checkpoint))
    except ValueError: report['wrong_method_specification_rejected'] = True
    else: raise AssertionError('Method specification was not checked')
    adapter.specification.copy_(spec)
    report['strict_reload_readouts_and_state_reset'] = True
    report['legacy_hi1_unchanged'] = legacy_check(args)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('P3_A_CHECK_OK '+json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained',default='')
    p.add_argument('--real-data',default='')
    p.add_argument('--output',default=str(ROOT/'work/p3_a_check'))
    run(p.parse_args())
