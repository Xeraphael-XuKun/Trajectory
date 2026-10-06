"""Optional real-image CLIP/AMP smoke; never launches an epoch or saves weights."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from config import cfg
from datasets import make_dataloader
from model import make_model
from model.c0_supervision import VARIANTS
from loss import make_loss
from loss.c0_supervision import c0_aux_loss, condition_masks
from solver import make_optimizer
from utils.runtime import configure_cudnn


def main():
    parser = argparse.ArgumentParser(description='C0辅助监督真实图像单batch检查，不替代完整训练')
    parser.add_argument('--config_file', default='configs/c0_aux_R1.yml')
    parser.add_argument('--variant', choices=('all',) + VARIANTS, default='all')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='0沿用配置；本地显存不足时可减小，仅用于smoke')
    parser.add_argument('--instances', type=int, default=0)
    parser.add_argument('--amp-init-scale', type=float, default=1024.0,
                        help='仅smoke的初始AMP scale；正式训练保持原GradScaler默认值')
    parser.add_argument('--output', required=True, help='独立JSON结果路径，不覆盖已有文件')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    c = cfg.clone()
    c.merge_from_file(args.config_file)
    c.merge_from_list(args.opts)
    if args.batch_size:
        c.SOLVER.IMS_PER_BATCH = args.batch_size
    if args.instances:
        c.DATALOADER.NUM_INSTANCE = args.instances
    c.DATALOADER.NUM_WORKERS = 0
    torch.set_num_threads(4)
    random.seed(c.SOLVER.SEED)
    np.random.seed(c.SOLVER.SEED)
    torch.manual_seed(c.SOLVER.SEED)
    torch.cuda.manual_seed_all(c.SOLVER.SEED)
    configure_cudnn(c.SOLVER.CUDNN_BENCHMARK)
    loader, _, _, _, nclasses, cameras, views = make_dataloader(c)
    # Smoke must exercise modality AND platform relations. Select a real PKM
    # batch with support, without altering the formal training sampler.
    for batch_index, (images, target, cams) in enumerate(loader):
        ids = target.repeat(len(images))
        cameras_flat = torch.cat(cams)
        mods = torch.arange(len(images)).repeat_interleave(len(target))
        same_id, cross_cam, same_cell, groups = condition_masks(ids, cameras_flat, mods)
        valid_negative = ((~same_id & cross_cam)[:, None, :] & same_cell[None, :, :]).any(2)
        coverage = [(same_id & cross_cam & group & valid_negative).any().item() for group in groups]
        if all(coverage):
            break
        if batch_index >= 15:
            raise RuntimeError('前16个真实batch未覆盖四类关系；增大smoke batch，不修改正式sampler')
    images = [x.cuda() for x in images]
    cams = [x.cuda() for x in cams]
    target = target.cuda()
    repeated = target.repeat(len(images))
    model = make_model(c, nclasses, cameras, views).cuda().train()
    loss_fn, center = make_loss(c, nclasses)
    initial_bn = copy.deepcopy(model.bottleneck.state_dict())
    # All variants are parameter-free, so one initial model suffices for independent
    # forward/backward checks. Only the final variant performs an optimizer step.
    variants = VARIANTS if args.variant == 'all' else (args.variant,)
    results = []
    for index, variant in enumerate(variants):
        model.c0_aux_variant = variant
        settings = c.C0_AUX.clone()
        settings.VARIANT = variant
        model.bottleneck.load_state_dict(initial_bn)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        with torch.no_grad(), torch.cuda.amp.autocast():
            ordinary = model(images, target, cams)
        reference_rng = torch.cuda.get_rng_state()
        reference_bn = copy.deepcopy(model.bottleneck.state_dict())
        model.bottleneck.load_state_dict(initial_bn)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.cuda.amp.autocast():
            out = model(images, target, cams, c0_probe_step=0)
            main_loss, _, _ = loss_fn(out[0], out[1], repeated)
            aux_loss, stats = c0_aux_loss(out[3], repeated, settings)
        assert torch.equal(reference_rng, torch.cuda.get_rng_state())
        for expected, actual in zip(ordinary, out[:3]):
            assert torch.equal(expected, actual), 'main output changed'
        for key, value in reference_bn.items():
            assert torch.equal(value, model.bottleneck.state_dict()[key]), 'BN updated by probe'
        assert torch.isfinite(aux_loss), 'nonfinite aux loss'
        scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_init_scale)
        scaler.scale(aux_loss).backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if name != 'base.token_trajectory.gain':
                assert parameter.grad is None, 'aux gradient escaped: ' + name
        aux_grad = model.base.token_trajectory.gain.grad.detach() / scaler.get_scale()
        assert torch.isfinite(aux_grad).all(), 'nonfinite gain gradient'
        norms = aux_grad.flatten(1).norm(dim=1).cpu().tolist()
        model.zero_grad(set_to_none=True)
        scaler.scale(main_loss + settings.WEIGHT * aux_loss).backward()
        stepped = index == len(variants) - 1
        if stepped:
            optimizer, _ = make_optimizer(c, model, center)
            scaler.unscale_(optimizer)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            before = model.base.token_trajectory.gain.detach().clone()
            scaler.step(optimizer)
            scaler.update()
            assert not torch.equal(before, model.base.token_trajectory.gain), 'optimizer did not update gain'
            del optimizer
        else:
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.cuda.synchronize()
        item = {'variant': variant, 'main_loss': float(main_loss.detach()),
                'aux_loss': float(aux_loss.detach()), 'gain_aux_grad_norm_by_row': norms,
                'valid_anchors_by_probe': {str(k): v[0][:, 0].cpu().tolist() for k, v in stats.items()},
                'main_outputs_rng_bn_equal': True, 'aux_gain_only': True,
                'combined_backward_finite': True, 'optimizer_step': stepped,
                'smoke_seconds': time.perf_counter() - started,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3}
        results.append(item)
        print(json.dumps(item), flush=True)
        del out, ordinary, main_loss, aux_loss, stats, aux_grad
        model.zero_grad(set_to_none=True)
    report = {'status': 'SMOKE_OK_NOT_FULL_TRAINING', 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'logical_batch': c.SOLVER.IMS_PER_BATCH,
              'images_per_batch': len(repeated), 'instances': c.DATALOADER.NUM_INSTANCE,
              'smoke_amp_init_scale': args.amp_init_scale,
              'selected_real_batch_index': batch_index,
              'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('C0_AUX_REAL_BATCH_SMOKE_OK', flush=True)


if __name__ == '__main__':
    main()
