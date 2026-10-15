"""P1 formula, isolation, real-model gradients and checkpoint checks; no training run."""
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
from model.backbones.history_p1 import HistoryP1Adapter
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from utils.history_training import history_gradient_groups, clip_history_gradients
from check_history import historical_module


def seed(value=1234):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def norm(parameters):
    return sum(float(p.grad.detach().abs().sum()) for p in parameters if p.grad is not None)


def formula_check():
    seed(1235)
    a = HistoryP1Adapter(16, 4, 4, rank=4)
    h = torch.randn(2, 5, 16, requires_grad=True)
    r, hist, previous, first = a.correction(h, 0, [], None, True, True)
    assert torch.count_nonzero(r) == 0 and first['prediction'] is None
    assert not hist[0].requires_grad and not previous.requires_grad
    assert len(a.anchors) == len(a.predictors) == len(a.correctors) == 3
    assert a.gain.shape == (3, 1, 4, 16)
    assert all(torch.count_nonzero(p[-1].weight) == 0 and torch.count_nonzero(p[-1].bias) == 0 for p in a.predictors)
    assert torch.allclose(a.coordinate.T @ a.coordinate, torch.eye(4), atol=1e-6)
    with torch.no_grad():
        a.gain.copy_(torch.linspace(-.1, .1, a.gain.numel()).reshape_as(a.gain))
        a.predictors[0][-1].bias.fill_(.13)  # nonzero prediction must be detached too.
    h2 = torch.randn_like(h, requires_grad=True)
    r, hist2, previous2, item = a.correction(h2, 1, hist, previous, True, True)
    s = F.layer_norm(h2[:, 1:], (16,), eps=1e-5) @ a.coordinate
    prediction = a.predictors[0](torch.cat((hist[-1], torch.zeros_like(hist[-1])), -1))
    e = s - (hist[-1] + prediction).detach()
    d = a.anchors[0](e) + a.correctors[0](torch.cat((s, e, previous), -1))
    expected = .25 * a.gain[0] * F.layer_norm(d, (16,), eps=1e-5)
    assert torch.equal(r[:, 1:], expected) and torch.count_nonzero(r[:, :1]) == 0
    assert torch.equal(hist2[-1], s.detach())
    assert torch.equal(previous2, r[:, 1:].detach() @ a.coordinate)
    task = (r * torch.randn_like(r)).sum()
    task.backward(retain_graph=True)
    assert norm(a.predictors.parameters()) == 0 and norm(a.correctors.parameters()) > 0
    assert h2.grad is not None and h.grad is None
    a.zero_grad(set_to_none=True); h2.grad = None
    aux = {'prediction_layers': [2], 'layers': {2: item}}
    loss = history_prediction_loss(aux)
    loss.backward()
    assert norm(a.predictors.parameters()) > 0 and norm(a.correctors.parameters()) == 0
    assert a.gain.grad is None and h2.grad is None
    # Third layer reads the actual previous action and old state increment.
    r3, hist3, previous3, item3 = a.correction(torch.randn_like(h), 2, hist2, previous2, True)
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
    report = {'status': 'LOCAL_CHECK_ONLY', 'torch': torch.__version__, 'formula_and_gradient_boundaries': True,
              'separate_clipping': True, 'formal_training': False, 'source_base': '30eaccc'}
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_P1.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
    if args.pretrained:
        c.MODEL.PRETRAIN_PATH = args.pretrained
    baseline_cfg = c.clone(); baseline_cfg.HISTORY.ENABLED = False
    old = historical_module('model/make_model.py', 'model.p1_reference_model', '30eaccc')
    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    for mode in ('baseline', 'c0'):
        reference_cfg = baseline_cfg.clone()
        reference_cfg.MODEL.TOKEN_TRAJECTORY = mode == 'c0'
        reference_cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX = 0.0
        seed(); reference = old.make_model(reference_cfg, 500, 7, 2)
        seed(); current = make_model(reference_cfg, 500, 7, 2)
        assert reference.state_dict().keys() == current.state_dict().keys()
        assert all(torch.equal(v, current.state_dict()[k]) for k, v in reference.state_dict().items())
        if mode == 'c0':
            with torch.no_grad():
                reference.base.token_trajectory.gain.fill_(.002)
                current.base.token_trajectory.gain.fill_(.002)
        with torch.no_grad():
            seed(); left = reference(images)
            seed(); right = current(images)
        assert all(torch.equal(x, y) for x, y in zip(left, right))
        old_solver = historical_module('solver/make_optimizer.py', 'solver.p1_reference_optimizer', '30eaccc')
        old_opt, _ = old_solver.make_optimizer(reference_cfg, reference, nn.Linear(1, 1))
        new_opt, _ = make_optimizer(reference_cfg, current, nn.Linear(1, 1))
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
    adapter = model.base.history_adapter
    assert sum(p.numel() for p in adapter.parameters()) == 2443584
    assert model.base.token_trajectory is None
    seed(1235); projected, _ = torch.linalg.qr(torch.randn(768, 64), mode='reduced')
    assert torch.equal(projected, adapter.coordinate)
    optimizer, _ = make_optimizer(c, model, nn.Linear(1, 1))
    group_by_id = {id(p): g for g in optimizer.param_groups for p in g['params']}
    for name, p in adapter.named_parameters():
        family = name.split('.')[0]
        g = group_by_id[id(p)]
        assert g['lr'] == (3.5e-4 if family == 'gain' else 1e-4)
        assert g['weight_decay'] == (1e-4 if family == 'predictors' else 0)
    assert group_by_id[id(model.base.blocks[0].attn.qkv.weight)]['lr'] == 5e-6
    groups = history_gradient_groups(model)
    assert not ({id(p) for p in groups['task']} & {id(p) for p in groups['prediction']})
    assert sum(len(v) for v in groups.values()) == sum(p.requires_grad for p in model.parameters())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        seed(); out = model(images, history_diagnostics=True)
        task = F.cross_entropy(out[0], target) + TripletLoss()(out[1], target)[0]
        prediction = history_prediction_loss(out[3])
        task.backward(retain_graph=True)
        assert norm(adapter.predictors.parameters()) == 0
        assert norm([adapter.gain]) > 0
        assert (norm(adapter.correctors.parameters()) > 0) == (step > 0)
        prediction.backward()
        assert norm(adapter.predictors.parameters()) > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clip_history_gradients(groups)
        optimizer.step()
    report['parameter_count_projection_lr_wd_and_real_updates'] = True
    report['adapter_parameters'] = sum(p.numel() for p in adapter.parameters())
    model.zero_grad(set_to_none=True)
    del optimizer, out, task, prediction, groups; gc.collect()
    model.eval()
    with torch.no_grad():
        first = model(images[0], mode=1)
        model(images[1], mode=2)
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
    with torch.no_grad():
        assert torch.equal(first, model(images[0], mode=1))
    assert not torch.equal(pre, first)
    # Baseline/legacy checkpoint cannot silently initialize a P1 adapter.
    expected_spec = adapter.specification.clone()
    with torch.no_grad(): adapter.specification[0] = -1
    try:
        model.load_param(str(checkpoint))
    except ValueError:
        report['wrong_spec_rejected'] = True
    else:
        raise AssertionError('P1 checkpoint specification was not checked')
    adapter.specification.copy_(expected_spec)
    report['disk_reload_readouts_and_forward_state_reset'] = True
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('P1_CHECK_OK ' + json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--real-data', default='')
    parser.add_argument('--output', default=str(ROOT / 'work/p1_check'))
    run(parser.parse_args())
