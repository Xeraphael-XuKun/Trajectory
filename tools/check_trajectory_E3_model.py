"""Real CLIP/YACS/model/optimizer/reload integration on CPU (not GPU acceptance)."""
import argparse
import gc
import importlib.util
import subprocess
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


def check_uniform_e2_control(e2_root):
    """Load only E2's actuator as a test reference; never register it for training."""
    from model.backbones.token_trajectory_round5 import Round5VelocityTrajectory
    source = e2_root / 'model/backbones/token_trajectory_round5.py'
    spec = importlib.util.spec_from_file_location('model.backbones._e2_uniform_control', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    e2_cfg = cfg.clone()
    e2_cfg.merge_from_file(str(e2_root / 'configs/trajectory_E2_attn_transport.yml'))
    e3_cfg = cfg.clone()
    e3_cfg.merge_from_file(str(ROOT / 'configs/trajectory_E3_uniform_transport.yml'))
    e2_cfg.MODEL.TOKEN_TRAJECTORY_VARIANT = e3_cfg.MODEL.TOKEN_TRAJECTORY_VARIANT
    e2_cfg.OUTPUT_DIR = e3_cfg.OUTPUT_DIR
    assert e2_cfg.dump() == e3_cfg.dump()
    torch.manual_seed(1234)
    results = []
    for depth, tokens, channels in [(4, 5, 8), (12, 129, 768)]:
        e2 = reference.Round5VelocityTrajectory(depth, tokens, channels, 'velocity_attn_transport')
        e3 = Round5VelocityTrajectory(depth, tokens, channels, 'velocity_uniform_transport')
        e3.gain.data.normal_(std=.1)
        e2.gain.data.copy_(e3.gain)
        assert e2.trajectory_parameters == e3.trajectory_parameters
        assert e2.method_signature[0] == 702 and e3.method_signature[0] == 703
        v2 = torch.randn(2, tokens, channels, requires_grad=True)
        v3 = v2.detach().clone().requires_grad_()
        p = torch.full((2, tokens, tokens), 1/tokens, requires_grad=True)
        target = torch.randn_like(v2)
        max_output_error = 0.0
        for i in range(1, depth):
            r2 = e2(i, v2, previous_attention=p)
            r3 = e3(i, v3)
            torch.testing.assert_close(r2, r3, rtol=1e-5, atol=1e-6)
            max_output_error = max(max_output_error, (r2-r3).abs().max().item())
            (r2*target).sum().backward()
            (r3*target).sum().backward()
        assert p.grad is None
        torch.testing.assert_close(v2.grad, v3.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(e2.gain.grad, e3.gain.grad, rtol=1e-5, atol=1e-6)
        results.append({'shape':[depth, tokens, channels], 'sites':depth-1,
                        'max_output_abs_error':max_output_error,
                        'max_velocity_grad_abs_error':(v2.grad-v3.grad).abs().max().item(),
                        'max_gain_grad_abs_error':(e2.gain.grad-e3.gain.grad).abs().max().item()})
    return {'status':'PASSED', 'e2_root':str(e2_root),
            'e2_commit':subprocess.check_output(['git','-C',str(e2_root),'rev-parse','HEAD']).decode().strip(),
            'e2_worktree_status':subprocess.check_output(['git','-C',str(e2_root),'status','--short']).decode().strip(),
            'merged_recipe_parity':'only variant and output differ', 'device':'cpu', 'dtype':'float32',
            'e2_attention_grad':None, 'cases':results}


def main():
    parser = argparse.ArgumentParser(description='E3 真实 CLIP 的 CPU 构造/优化器/重载检查')
    parser.add_argument('--clip-path', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--e2-root', type=Path, help='可选：已实现 E2 的源码，用于真实类的均匀 P 对照')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    uniform_control = check_uniform_e2_control(args.e2_root.resolve()) if args.e2_root else None
    c = cfg.clone()
    c.merge_from_file(str(ROOT / 'configs/trajectory_E3_uniform_transport.yml'))
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
    path = args.output_dir / 'integration_E3.pth'
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
              'input': 'synthetic images', 'uniform_e2_control':uniform_control, 'formal_training': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
