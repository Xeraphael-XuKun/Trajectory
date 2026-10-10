"""E3 真实权重/数据短检查；不保存训练权重，不接续正式训练。"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from model.backbones.token_trajectory_variants import validate_trajectory_config
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.runtime import configure_cudnn, runtime_summary


def main():
    parser = argparse.ArgumentParser(description='E3 真实图像 AMP 短检查，不替代正式训练')
    parser.add_argument('--config_file', default=str(ROOT / 'configs/trajectory_E3_uniform_transport.yml'))
    parser.add_argument('--clip-path', required=True)
    parser.add_argument('--data-root', required=True, help='包含 WHU-MARS 子目录的父目录')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--logical-batch', type=int, default=8)
    parser.add_argument('--amp-init-scale', type=float, default=128,
                        help='仅短检查使用；正式 train.py 的 GradScaler 默认值保持原样')
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('不覆盖既有检查目录：{}'.format(out))
    if args.steps < 1 or args.logical_batch < 8 or args.logical_batch % 4:
        raise ValueError('steps 至少1；逻辑 batch 为4的倍数且至少8，保证有负身份')
    if not torch.cuda.is_available():
        raise RuntimeError('该检查需要 CUDA；CPU 构造检查不能替代 AMP 前反向')
    c = cfg.clone()
    c.merge_from_file(args.config_file)
    c.MODEL.PRETRAIN_PATH = args.clip_path
    c.DATASETS.ROOT_DIR = args.data_root
    c.SOLVER.IMS_PER_BATCH = args.logical_batch
    c.DATALOADER.NUM_WORKERS = 0
    c.OUTPUT_DIR = str(out)
    if c.MODEL.TOKEN_TRAJECTORY_VARIANT != 'velocity_uniform_transport':
        raise ValueError('该工具仅检查 E3')
    validate_trajectory_config(c)
    out.mkdir(parents=True)
    (out / 'config.yml').write_text(c.dump(), encoding='utf-8')
    random.seed(c.SOLVER.SEED)
    np.random.seed(c.SOLVER.SEED)
    torch.manual_seed(c.SOLVER.SEED)
    torch.cuda.manual_seed_all(c.SOLVER.SEED)
    configure_cudnn(c.SOLVER.CUDNN_BENCHMARK)
    loader, _, _, _, classes, cameras, views = make_dataloader(c)
    model = make_model(c, num_class=classes, camera_num=cameras, view_num=views).cuda().train()
    loss_fn, center = make_loss(c, num_classes=classes)
    optimizer, _ = make_optimizer(c, model, center)
    gain = model.base.token_trajectory.gain
    assert gain.shape == (11, 1, 129, 768) and gain.numel() == 1089792
    groups = [g for g in optimizer.param_groups if any(p is gain for p in g['params'])]
    assert len(groups) == 1 and (groups[0]['lr'], groups[0]['weight_decay']) == (.00035, .0001)
    scheduler = create_scheduler(c, optimizer, iters_per_epoch=len(loader))
    scheduler.step(1)
    scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_init_scale)
    rows = []
    observations = []
    def observe(module, inputs, output):
        observations.append({'layer': inputs[0], 'velocity_requires_grad': inputs[1].requires_grad})
    hook = model.base.token_trajectory.register_forward_hook(observe)
    torch.cuda.reset_peak_memory_stats()
    for step, (images, ids, cams) in enumerate(loader):
        scheduler.step_update(step)
        assert ids.unique().numel() >= 2
        images = [x.cuda() for x in images]
        ids, cams = ids.cuda(), [x.cuda() for x in cams]
        optimizer.zero_grad()
        observations.clear()
        start = time.perf_counter()
        with torch.cuda.amp.autocast():
            scores, pre, post = model(images, ids, cams)
            loss, ce, triplet = loss_fn(scores, pre, ids.repeat(len(images)))
        assert torch.isfinite(loss) and torch.isfinite(pre).all() and torch.isfinite(post).all()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gain_finite = bool(torch.isfinite(gain.grad).all())
        row_norms = gain.grad.float().flatten(1).norm(dim=1).detach().cpu().tolist()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        qk_grad = model.base.blocks[0].attn.qkv.weight.grad[:1536]
        qk_finite = bool(torch.isfinite(qk_grad).all())
        qk_nonzero = bool(qk_grad.abs().sum() > 0)
        before = gain.detach().clone()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        row = {'step': step + 1, 'ce': ce.item(), 'triplet': triplet.item(),
               'loss': loss.item(), 'gain_grad_finite': gain_finite,
               'gain_grad_norm_by_layer': row_norms,
               'gain_lr': groups[0]['lr'],
               'qk_grad_finite': qk_finite, 'qk_grad_nonzero': qk_nonzero,
               'transport_observations': list(observations),
               'grad_norm': float(grad_norm) if torch.isfinite(grad_norm) else None,
               'grad_norm_finite': bool(torch.isfinite(grad_norm)),
               'gain_changed': not torch.equal(before, gain),
               'scale_before': scale_before, 'scale_after': scaler.get_scale(),
               'seconds': time.perf_counter() - start}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if len(rows) == args.steps:
            break
    hook.remove()
    checks = {
        'steps_complete': len(rows) == args.steps,
        'all_velocity_observations_differentiable': all(
            [v['layer'] for v in r['transport_observations']] == list(range(1, 12)) and
            all(v['velocity_requires_grad'] for v in r['transport_observations']) for r in rows),
        'original_qk_gradients': all(r['qk_grad_finite'] and r['qk_grad_nonzero'] for r in rows),
        'all_gradients_finite': all(r['gain_grad_finite'] and r['grad_norm_finite'] for r in rows),
        'first_step_all_gain_rows_receive_gradient': all(v > 0 for v in rows[0]['gain_grad_norm_by_layer']),
        'gain_updates': any(r['gain_changed'] for r in rows) and bool(torch.count_nonzero(gain) > 0),
        'one_bn_update_per_step': model.bottleneck.num_batches_tracked.item() == args.steps,
    }
    model.eval()
    with torch.no_grad():
        selected = model(images[0], camids=cams[0], mode=1)
    assert selected.shape == (args.logical_batch, 768) and torch.isfinite(selected).all()
    result = {'status': 'PASSED' if all(checks.values()) else 'FAILED',
              'checks': checks, 'runtime': runtime_summary(), 'python': sys.executable,
              'config_file': str(Path(args.config_file).resolve()),
              'logical_batch': args.logical_batch, 'actual_images': sum(len(x) for x in images),
              'amp_init_scale_smoke_only': args.amp_init_scale, 'rows': rows,
              'signature': model.base.token_trajectory.method_signature.cpu().tolist(),
              'gain_nonzero': torch.count_nonzero(gain).item(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'formal_training': False}
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise RuntimeError('E3 smoke 未通过，完整记录见 summary.json')


if __name__ == '__main__':
    main()
