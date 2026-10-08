"""Meaningful local checks; no 60-epoch experiment and no forced launch gate.

Run from the repository root. --pretrained adds the real CLIP loading test.
--real-data checks one batch from the unchanged sampler / transforms.
"""
import argparse
import gc
import importlib.util
import json
import random
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from config import cfg
from model import make_model
from model.backbones.history_adapter import HistoryInnovationAdapter
from loss.history_prediction import history_prediction_loss, history_diagnostics
from loss.triplet_loss import TripletLoss
from solver import make_optimizer


def seed():
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)


def historical_module(file, name, ref):
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', ref + ':' + file]).decode('utf-8')
    module = types.ModuleType(name)
    module.__package__ = name.rsplit('.', 1)[0]
    exec(compile(source, file + '@' + ref, 'exec'), module.__dict__)
    return module


def require_nonzero(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None) > 0


def run(args):
    torch.set_num_threads(4)
    report = {'torch': torch.__version__, 'device': 'cpu', 'formal_training': False}
    configs = {}
    for name in ('baseline', 'HI0', 'HI1'):
        c = cfg.clone(); c.merge_from_file(str(ROOT / ('configs/history_' + name + '.yml')))
        c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
        if args.pretrained: c.MODEL.PRETRAIN_PATH = args.pretrained
        configs[name] = c
    c = configs['HI1']
    # Small per-token formula and gradient boundary checks.
    seed()
    a = HistoryInnovationAdapter(16, 3, 4, rank=4, prediction_layers=[2, 3])
    h = torch.randn(2, 5, 16, requires_grad=True)
    r, history, previous, item = a.correction(h, 0, [], None, True, True)
    assert torch.count_nonzero(r) == 0 and not history[-1].requires_grad and not previous.requires_grad
    with torch.no_grad(): a.gain.fill_(0.1)
    r, history2, previous2, item = a.correction(h * 1.1, 1, history, previous, True, True)
    assert torch.count_nonzero(r[:, 0]) == 0 and torch.count_nonzero(r[:, 1:]) > 0
    expected_s = torch.nn.functional.layer_norm((h * 1.1)[:, 1:], (16,)) @ a.coordinate
    pred = a.predictors[0](torch.cat((history[-1], torch.zeros_like(history[-1])), -1))
    e = expected_s - (history[-1] + pred)
    d = a.anchors[1](e) + a.correctors[1](torch.cat((expected_s, e, previous), -1))
    assert torch.equal(r[:, 1:], a.alpha * a.gain[1].tanh() * d)
    assert torch.allclose(a.coordinate.T @ a.coordinate, torch.eye(4), atol=1e-6)
    pred_loss = history_prediction_loss({'layers': {2: item}, 'prediction_layers': [2]})
    pred_loss.backward()
    assert h.grad is None and a.gain.grad is None
    assert require_nonzero(a.predictors.parameters())
    assert not require_nonzero(a.anchors.parameters())
    report['formula_and_detached_prediction_gradient'] = True

    # Historical backbone and wrapper are loaded from the exact source commit.
    vb = historical_module('model/backbones/vit_pytorch.py', 'model.backbones.reference_vit', args.reference)
    mm = historical_module('model/make_model.py', 'model.reference_make_model', args.reference)
    mm.vit_base_clip = vb.vit_base_clip
    mm.__dict__['__factory_T_type']['vit_base_clip'] = vb.vit_base_clip
    seed(); reference = mm.make_model(configs['baseline'], 500, 7, 2)
    seed(); baseline = make_model(configs['baseline'], 500, 7, 2)
    assert reference.state_dict().keys() == baseline.state_dict().keys()
    assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in reference.state_dict().items())
    reference.train(); baseline.train()
    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    with torch.no_grad():
        seed(); ref_out = reference(images)
        seed(); base_out = baseline(images)
    assert all(torch.equal(x, y) for x, y in zip(ref_out, base_out))
    report['disabled_equals_reference'] = True
    # C0 path must still agree with the same historical implementation.
    from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
    for m in (reference, baseline):
        m.base.token_trajectory = DenseCrossLayerTokenTrajectory(12, 129, 768, 0.0)
        with torch.no_grad(): m.base.token_trajectory.gain.fill_(0.002)
    with torch.no_grad():
        seed(); ref_c0 = reference(images)
        seed(); base_c0 = baseline(images)
    assert all(torch.equal(x, y) for x, y in zip(ref_c0, base_c0))
    report['historical_c0_path_equal'] = True
    del reference, baseline; gc.collect()

    seed(); baseline = make_model(configs['baseline'], 500, 7, 2)
    baseline_rng = torch.random.get_rng_state().clone()
    seed(); model = make_model(c, 500, 7, 2)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
    assert model.base.token_trajectory is None
    with torch.no_grad():
        seed(); base_out = baseline(images)
        seed(); out = model(images, history_diagnostics=True)
    assert all(torch.equal(x, y) for x, y in zip(base_out, out[:3]))
    report['zero_gain_equals_baseline_and_rng'] = True
    del baseline; gc.collect()
    if args.real_data:
        from datasets import make_dataloader
        data_cfg = c.clone(); data_cfg.DATASETS.ROOT_DIR = args.real_data
        data_cfg.DATALOADER.NUM_WORKERS = 0
        data_cfg.DATALOADER.NUM_INSTANCE = 2; data_cfg.SOLVER.IMS_PER_BATCH = 4
        loader, _, _, _, _, _, _ = make_dataloader(data_cfg)
        seed(); images, labels, _ = next(iter(loader))
        report['real_batch'] = {'modalities': len(images), 'images_per_modality': len(labels), 'identities': labels.unique().numel()}
    else:
        labels = torch.tensor([0, 1])
    target = labels.repeat(len(images))
    model.train(); seed(); out = model(images, history_diagnostics=True)
    raw = history_prediction_loss(out[3])
    ce = torch.nn.functional.cross_entropy(out[0], target)
    tri = TripletLoss()(out[1], target)[0]
    criterion = nn.Linear(1, 1)  # center optimizer is unused by this baseline.
    optimizer, _ = make_optimizer(c, model, criterion)
    by_id = {id(p): group for group in optimizer.param_groups for p in group['params']}
    assert all(by_id[id(p)]['lr'] == c.SOLVER.BASE_LR for p in model.base.history_adapter.parameters())
    assert by_id[id(model.base.blocks[0].attn.qkv.weight)]['lr'] == c.SOLVER.PRETRAINED_LR
    (ce + tri + raw).backward()
    adapter = model.base.history_adapter
    assert require_nonzero([adapter.gain]) and require_nonzero(adapter.predictors.parameters())
    assert not require_nonzero(adapter.correctors.parameters())  # expected at zero gain.
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad()
    assert torch.count_nonzero(adapter.gain) > 0
    seed(); out2 = model(images)
    (torch.nn.functional.cross_entropy(out2[0], target) + TripletLoss()(out2[1], target)[0]).backward()
    assert require_nonzero(adapter.correctors.parameters()) and require_nonzero(adapter.predictors.parameters())
    report['real_model_gradient_and_optimizer'] = True
    report['first_step_losses'] = {'ce': float(ce.detach()), 'triplet': float(tri.detach()), 'prediction': float(raw.detach())}
    report['adapter_parameters'] = sum(p.numel() for p in adapter.parameters())
    model.zero_grad(); del optimizer, out, out2; gc.collect()
    model.eval()
    with torch.no_grad():
        first = model(images[0], mode=1)
        model(images[1], mode=2)
        second = model(images[0], mode=1)
    assert torch.equal(first, second)
    path = ROOT / 'work/history_check/checkpoint.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    with torch.no_grad(): adapter.gain.add_(1); adapter.coordinate.zero_()
    model.load_param(str(path))
    with torch.no_grad():
        reloaded = model(images[0], mode=1)
        model.neck_feat = 'before'; pre = model(images[0], mode=1)
        direct = model.base(images[0])
    assert torch.equal(first, reloaded) and torch.equal(pre, direct)
    report['state_reset_disk_reload_both_readouts'] = True
    assert not torch.equal(pre, reloaded)
    del model; gc.collect()
    # Same architecture/init in HI0; only the training objective differs.
    c0, c1 = configs['HI0'].clone(), configs['HI1'].clone()
    c0.HISTORY.LOSS_WEIGHT = c1.HISTORY.LOSS_WEIGHT
    c0.OUTPUT_DIR = c1.OUTPUT_DIR
    assert c0.dump() == c1.dump()
    report['hi0_hi1_single_factor_config'] = True
    (ROOT / 'work/history_check/report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('HISTORY_CHECK_OK ' + json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', default='')
    p.add_argument('--real-data', default='')
    p.add_argument('--reference', default='1e8440f')
    run(p.parse_args())
