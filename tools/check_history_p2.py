"""P2 spatial formula, gradient boundaries, baseline isolation and strict reload."""
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
from model.backbones.history_p2 import HistoryP2Adapter
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from utils.history_training import history_gradient_groups, clip_history_gradients
from check_history import historical_module


DIRECTION_FAMILIES = ('fusions', 'depthwise3', 'depthwise5', 'up_projections')


def seed(value=1234):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def norm(parameters):
    return sum(float(p.grad.detach().abs().sum()) for p in parameters if p.grad is not None)


def formula_check():
    seed(1235)
    a = HistoryP2Adapter(16, 4, 12, rank=4, grid_size=(3, 4))
    h = torch.randn(2, 13, 16, requires_grad=True)
    r, hist, previous, first = a.correction(h, 0, [], None, True, True)
    assert torch.count_nonzero(r) == 0 and first['prediction'] is None
    assert not hist[0].requires_grad and not previous.requires_grad
    assert a.gain.shape == (3, 1, 12, 16)
    assert not hasattr(a, 'anchors') and not hasattr(a, 'correctors')
    assert all(len(getattr(a, family)) == 3 for family in DIRECTION_FAMILIES)
    assert all(torch.count_nonzero(p[-1].weight) == 0 and torch.count_nonzero(p[-1].bias) == 0 for p in a.predictors)
    assert torch.allclose(a.coordinate.T @ a.coordinate, torch.eye(4), atol=1e-6)
    assert all(torch.count_nonzero(c.weight) == 0 and c.bias is None and c.groups == 4
               for c in list(a.depthwise3) + list(a.depthwise5))
    with torch.no_grad():
        a.gain.copy_(torch.linspace(-.1, .1, a.gain.numel()).reshape_as(a.gain))
        a.predictors[0][-1].bias.fill_(.13)
    h2 = torch.randn_like(h, requires_grad=True)
    zero_r, _, _, zero_item = a.correction(h2, 1, hist, previous, True, True)
    s = F.layer_norm(h2[:, 1:], (16,), eps=1e-5) @ a.coordinate
    prediction = a.predictors[0](torch.cat((hist[-1], torch.zeros_like(hist[-1])), -1))
    e = s - (hist[-1] + prediction).detach()
    fused = e + F.gelu(a.fusions[0](torch.cat((s, e, previous), -1)))
    direct_direction = a.up_projections[0](F.gelu(fused))
    expected_zero = .25 * a.gain[0] * F.layer_norm(direct_direction, (16,), eps=1e-5)
    assert torch.equal(zero_r[:, 1:], expected_zero)
    assert torch.count_nonzero(zero_r[:, 1:]) > 0 and zero_item['diagnostics']['spatial_update_rms'] == 0
    # A nonzero-kernel check distinguishes parallel kernels from sequential ones,
    # and catches spatial transposition or deletion of E's direct connection.
    with torch.no_grad():
        a.depthwise3[0].weight.copy_(torch.linspace(-.04, .05, 36).reshape_as(a.depthwise3[0].weight))
        a.depthwise5[0].weight.copy_(torch.linspace(.02, -.03, 100).reshape_as(a.depthwise5[0].weight))
    grid = fused.reshape(2, 3, 4, 4).permute(0, 3, 1, 2)
    mix = grid + .5 * (F.conv2d(grid, a.depthwise3[0].weight, padding=1, groups=4)
                       + F.conv2d(grid, a.depthwise5[0].weight, padding=2, groups=4))
    restored = mix.permute(0, 2, 3, 1).reshape(2, 12, 4)
    direction = a.up_projections[0](F.gelu(restored))
    expected = .25 * a.gain[0] * F.layer_norm(direction, (16,), eps=1e-5)
    r, hist2, previous2, item = a.correction(h2, 1, hist, previous, True, True)
    torch.testing.assert_close(r[:, 1:], expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(zero_r, r) and torch.count_nonzero(r[:, :1]) == 0
    assert torch.equal(hist2[-1], s.detach()) and torch.equal(previous2, r[:, 1:].detach() @ a.coordinate)
    (r * torch.randn_like(r)).sum().backward(retain_graph=True)
    assert norm(a.predictors.parameters()) == 0
    assert all(norm(getattr(a, family).parameters()) > 0 for family in DIRECTION_FAMILIES)
    assert h2.grad is not None and h.grad is None
    a.zero_grad(set_to_none=True); h2.grad = None
    history_prediction_loss({'prediction_layers': [2], 'layers': {2: item}}).backward()
    assert norm(a.predictors.parameters()) > 0 and h2.grad is None and a.gain.grad is None
    assert all(norm(getattr(a, family).parameters()) == 0 for family in DIRECTION_FAMILIES)
    _, hist3, previous3, item3 = a.correction(torch.randn_like(h), 2, hist2, previous2, True)
    assert len(hist3) == 2 and not previous3.requires_grad
    assert torch.equal(item3['prediction'], a.predictors[1](torch.cat((hist2[-1], hist2[-1] - hist2[-2]), -1)))
    assert a.correction(torch.randn_like(h), 3, hist3, previous3, False)[3] is None
    p, q = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))
    p.grad, q.grad = torch.full_like(p, 100), torch.full_like(q, 200)
    clip_history_gradients({'task': [p], 'prediction': [q]})
    assert abs(float(p.grad.norm()) - 1) < 1e-5 and abs(float(q.grad.norm()) - 1) < 1e-5


def run(args):
    torch.set_num_threads(4)
    formula_check()
    report = {'status': 'LOCAL_CHECK_ONLY', 'torch': torch.__version__, 'source_base': '30eaccc',
              'spatial_formula_zero_and_nonzero_kernels': True, 'gradient_boundaries_and_separate_clipping': True,
              'formal_training': False}
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_P2.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
    if args.pretrained: c.MODEL.PRETRAIN_PATH = args.pretrained
    baseline_cfg = c.clone(); baseline_cfg.HISTORY.ENABLED = False
    old = historical_module('model/make_model.py', 'model.p2_reference_model', '30eaccc')
    old_solver = historical_module('solver/make_optimizer.py', 'solver.p2_reference_optimizer', '30eaccc')
    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    for mode in ('baseline', 'c0'):
        rc = baseline_cfg.clone(); rc.MODEL.TOKEN_TRAJECTORY = mode == 'c0'; rc.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX = 0.0
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
    seed(); baseline = make_model(baseline_cfg, 500, 7, 2)
    rng = torch.random.get_rng_state().clone()
    seed(); model = make_model(c, 500, 7, 2)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
    with torch.no_grad():
        seed(); left = baseline(images)
        seed(); right = model(images, history_diagnostics=True)
    assert all(torch.equal(x, y) for x, y in zip(left, right[:3]))
    assert right[3]['prediction_layers'] == list(range(2, 13))
    assert set(right[3]['layers']) == set(range(1, 13))
    expected_loss = torch.stack([F.smooth_l1_loss(right[3]['layers'][l]['prediction'], right[3]['layers'][l]['target']) for l in range(2, 13)]).mean()
    assert torch.equal(expected_loss, history_prediction_loss(right[3]))
    report['zero_gain_baseline_rng_and_loss_denominator'] = True
    del baseline, left, right; gc.collect()
    adapter = model.base.history_adapter
    # Read P1's committed class only for the matched initial-coordinate check.
    # No P1 module is installed or executed in the P2 training model.
    p1_source = historical_module('model/backbones/history_p1.py', 'model.backbones.p2_p1_reference', '7dee827')
    seed(1235); p1 = p1_source.HistoryP1Adapter(768, 12, 128)
    assert torch.equal(p1.coordinate, adapter.coordinate)
    assert all(torch.equal(v, adapter.predictors.state_dict()[k]) for k, v in p1.predictors.state_dict().items())
    report['projection_and_predictor_initialization_equal_p1'] = True
    del p1; gc.collect()
    assert sum(p.numel() for p in adapter.parameters()) == 1926848
    assert model.base.token_trajectory is None and (adapter.grid_h, adapter.grid_w) == (16, 8)
    assert not hasattr(adapter, 'anchors') and not hasattr(adapter, 'correctors')
    if args.real_data:
        from datasets import make_dataloader
        dc = c.clone(); dc.DATASETS.ROOT_DIR = args.real_data
        dc.DATALOADER.NUM_WORKERS = 0; dc.SOLVER.IMS_PER_BATCH = 4; dc.DATALOADER.NUM_INSTANCE = 2
        seed(); loader, _, _, _, classes, _, _ = make_dataloader(dc)
        images, labels, _ = next(iter(loader))
        assert classes == 500
        report['real_batch'] = {'actual_images': sum(len(x) for x in images), 'identities': labels.unique().numel()}
    else:
        labels = torch.tensor([0, 1])
    target = labels.repeat(3)
    optimizer, _ = make_optimizer(c, model, nn.Linear(1, 1))
    by_id = {id(p): g for g in optimizer.param_groups for p in g['params']}
    for name, p in adapter.named_parameters():
        family = name.split('.')[0]; group = by_id[id(p)]
        assert group['lr'] == (3.5e-4 if family == 'gain' else 1e-4)
        assert group['weight_decay'] == (1e-4 if family == 'predictors' else 0)
    assert by_id[id(model.base.blocks[0].attn.qkv.weight)]['lr'] == 5e-6
    groups = history_gradient_groups(model)
    assert not ({id(p) for p in groups['task']} & {id(p) for p in groups['prediction']})
    assert sum(len(v) for v in groups.values()) == sum(p.requires_grad for p in model.parameters())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        seed(); out = model(images, history_diagnostics=True)
        task = F.cross_entropy(out[0], target) + TripletLoss()(out[1], target)[0]
        prediction = history_prediction_loss(out[3])
        task.backward(retain_graph=True)
        assert norm(adapter.predictors.parameters()) == 0 and norm([adapter.gain]) > 0
        assert all((norm(getattr(adapter, family).parameters()) > 0) == (step > 0) for family in DIRECTION_FAMILIES)
        prediction.backward()
        assert norm(adapter.predictors.parameters()) > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clip_history_gradients(groups); optimizer.step()
    assert all(any(torch.count_nonzero(conv.weight) > 0 for conv in getattr(adapter, family)) for family in ('depthwise3', 'depthwise5'))
    report['parameter_count_lr_wd_and_direction_startup'] = True
    report['adapter_parameters'] = sum(p.numel() for p in adapter.parameters())
    model.zero_grad(set_to_none=True)
    del optimizer, out, task, prediction, groups; gc.collect()
    model.eval()
    with torch.no_grad():
        first = model(images[0], mode=1); model(images[1], mode=2)
        assert torch.equal(first, model(images[0], mode=1))
        model.neck_feat = 'before'; pre = model(images[0], mode=1)
        assert torch.equal(pre, model.base(images[0]))
        model.neck_feat = 'after'
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.pth'
    torch.save(model.state_dict(), checkpoint)
    with torch.no_grad():
        adapter.gain.add_(1); adapter.coordinate.zero_()
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    model.load_param(str(checkpoint))
    with torch.no_grad(): assert torch.equal(first, model(images[0], mode=1))
    assert not torch.equal(pre, first)
    expected_spec = adapter.specification.clone()
    with torch.no_grad(): adapter.specification[-2:] = torch.tensor([8., 16.])
    try:
        model.load_param(str(checkpoint))
    except ValueError:
        report['wrong_spatial_grid_spec_rejected'] = True
    else:
        raise AssertionError('P2 spatial grid specification was not checked')
    adapter.specification.copy_(expected_spec)
    report['strict_disk_reload_readouts_and_state_reset'] = True
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('P2_CHECK_OK ' + json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', default='')
    p.add_argument('--real-data', default='')
    p.add_argument('--output', default=str(ROOT / 'work/p2_check'))
    run(p.parse_args())
