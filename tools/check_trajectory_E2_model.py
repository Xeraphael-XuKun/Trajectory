"""Real CLIP/YACS/model/optimizer/reload integration on CPU (not GPU acceptance)."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from config import cfg
from model import make_model
from solver import make_optimizer
from utils.runtime import runtime_summary


def main():
    parser = argparse.ArgumentParser(description='E2 真实 CLIP 的 CPU 构造/优化器/重载检查')
    parser.add_argument('--clip-path', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    c = cfg.clone()
    c.merge_from_file(str(ROOT / 'configs/trajectory_E2_attn_transport.yml'))
    c.MODEL.PRETRAIN_PATH = args.clip_path
    baseline_cfg = c.clone()
    baseline_cfg.MODEL.TOKEN_TRAJECTORY = False
    baseline_cfg.MODEL.TOKEN_TRAJECTORY_VARIANT = 'dense'
    torch.manual_seed(1234)
    baseline = make_model(baseline_cfg, num_class=500, camera_num=7, view_num=2).eval()
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(1234)
    candidate = make_model(c, num_class=500, camera_num=7, view_num=2).eval()
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
    own = candidate.state_dict()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, own[key]), key
    gain = candidate.base.token_trajectory.gain
    assert tuple(gain.shape) == (11, 1, 129, 768) and gain.numel() == 1089792
    optimizer, _ = make_optimizer(c, candidate, nn.Linear(1, 1))
    group = [g for g in optimizer.param_groups if any(p is gain for p in g['params'])]
    assert len(group) == 1 and (group[0]['lr'], group[0]['weight_decay']) == (.00035, .0001)
    del optimizer
    image = torch.randn(2, 3, 256, 128)
    expected = {}
    for readout in ('before', 'after'):
        candidate.neck_feat = baseline.neck_feat = readout
        with torch.no_grad():
            expected[readout] = baseline(image, mode=1)
            assert torch.equal(expected[readout], candidate(image, mode=1))
    calls = []
    hook = candidate.base.token_trajectory.register_forward_pre_hook(
        lambda module, inputs: calls.append(inputs[0]))
    gain.data.normal_(std=.01)
    candidate.neck_feat = 'after'
    with torch.no_grad():
        prediction = candidate(image, mode=1)
    hook.remove()
    assert calls == list(range(1, 12))
    assert not torch.allclose(expected['after'], prediction)
    # Free the baseline before testing the actual training-checkpoint loader.
    del baseline, own
    gc.collect()
    path = args.output_dir / 'integration_E2.pth'
    torch.save(candidate.state_dict(), path)
    gain.data.zero_()
    candidate.bottleneck.running_mean.fill_(1)
    candidate.load_param(path)
    with torch.no_grad():
        assert torch.equal(prediction, candidate(image, mode=1))
    report = {'status': 'PASSED', 'device': 'cpu', 'runtime': runtime_summary(),
              'python': sys.executable, 'clip_path': args.clip_path,
              'gain_shape': list(gain.shape), 'parameters': gain.numel(),
              'gain_lr': group[0]['lr'], 'weight_decay': group[0]['weight_decay'],
              'injection_sites': calls, 'shared_initialization': 'bitwise',
              'initial_pre_post_output': 'bitwise', 'formal_loader_roundtrip': 'bitwise',
              'signature': candidate.base.token_trajectory.method_signature.tolist(),
              'input': 'synthetic images', 'formal_training': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
