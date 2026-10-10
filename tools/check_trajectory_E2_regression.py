"""Compare historical train/BN/readout/gradient/RNG against a fixed source tree."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [('baseline', 'dense', False, 0.0), ('T0', 'dense', True, 1.0),
          ('C0', 'dense', True, 0.0), ('T1', 'adaptive', True, 1.0),
          ('T2', 'split', True, 1.0), ('T4', 'token_gate', True, 1.0),
          ('P3', 'attention_velocity', True, 0.0),
          ('P4', 'mlp_velocity', True, 0.0)]


def capture(root, output, device):
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(root / 'tests'))
    import torch
    from torch.nn import functional as F
    from model.backbones.vit_pytorch import TransReID
    from loss.triplet_loss import TripletLoss
    from test_c0_supervision import wrapper
    torch.set_num_threads(2)
    records = {}
    for name, variant, enabled, mix in GROUPS:
        torch.manual_seed(1234)
        base = TransReID(img_size=(32, 16), patch_size=16, stride_size=16,
                        embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                        qkv_bias=True, drop_path_rate=.2, drop_rate=.1,
                        attn_drop_rate=.1, token_trajectory=enabled,
                        token_trajectory_variant=variant,
                        token_trajectory_accel_mix=mix)
        model = wrapper(base).to(device).train()
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        initial_rng = torch.random.get_rng_state()
        if enabled:
            with torch.no_grad():
                for p in model.base.token_trajectory.parameters():
                    p.normal_(std=.05)
        images = [torch.randn(4, 3, 32, 16, device=device) for _ in range(3)]
        ids = torch.tensor([0, 0, 1, 1], device=device)
        cams = [torch.tensor([0, 1, 5, 6], device=device) for _ in range(3)]
        out = model(images, ids, cams)
        loss = F.cross_entropy(out[0], ids.repeat(3)) + TripletLoss()(out[1], ids.repeat(3))[0]
        loss.backward()
        records[name] = {
            'initial': initial, 'initial_rng': initial_rng,
            'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'outputs': [v.detach().cpu() for v in out],
            'loss': loss.detach().cpu(), 'rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state() if device == 'cuda' else None,
            'gradients': {k: p.grad.detach().cpu().clone() if p.grad is not None else None
                          for k, p in model.named_parameters()},
        }
        model.eval()
        with torch.no_grad():
            model.neck_feat = 'before'
            records[name]['pre'] = model(images[0], camids=cams[0], mode=1).cpu()
            model.neck_feat = 'after'
            records[name]['post'] = model(images[0], camids=cams[0], mode=1).cpu()
    torch.save(records, output)


def main():
    parser = argparse.ArgumentParser(description='历史路径与固定源码逐项比较，不训练正式模型')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--capture-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_root is not None:
        capture(args.capture_root, args.output_dir, args.device)
        return
    if args.reference_root is None or not (args.reference_root / 'model').is_dir():
        parser.error('--reference-root must point to the fixed source snapshot')
    args.reference_root = args.reference_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # Preserve caller-supplied dependency paths, while each child imports its
    # own source tree first. No code from the other tree is reused via sys.modules.
    env['PYTHONPATH'] = os.pathsep.join(sys.path[1:])
    paths = []
    for label, root in [('reference', args.reference_root), ('candidate', ROOT)]:
        destination = args.output_dir / (label + '.pt')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--capture-root',
                        str(root), '--output-dir', str(destination), '--device', args.device],
                       env=env, check=True, cwd=str(root))
        paths.append(destination)
    import torch
    old, new = [torch.load(p, map_location='cpu') for p in paths]
    counts = {}

    def compare(a, b, key):
        if isinstance(a, dict):
            assert set(a) == set(b), key + ': keys changed'
            return sum(compare(a[k], b[k], key + '/' + k) for k in a)
        if isinstance(a, list):
            assert len(a) == len(b), key
            return sum(compare(x, y, key) for x, y in zip(a, b))
        if a is None:
            assert b is None, key
            return 0
        assert torch.equal(a, b), key + ': tensor changed'
        return 1

    for name, _, _, _ in GROUPS:
        counts[name] = compare(old[name], new[name], name)
    report = {'status': 'PASSED', 'comparison': 'bitwise', 'device': args.device,
              'reference_root': str(args.reference_root), 'candidate_root': str(ROOT),
              'checked_tensors': counts, 'formal_training': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
