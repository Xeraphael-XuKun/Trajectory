"""单组真实 batch 的数值/显存预检；不启动正式训练，不保存 checkpoint。"""
import argparse
import sys
from pathlib import Path
import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from solver import make_optimizer
from utils.runtime import prepare_run, runtime_info, write_json
from tools.verify_matrix import verify


def main():
    parser = argparse.ArgumentParser(description='八组实验的单卡数值预检')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    verify()
    cfg = load_config(args.config_file)
    original_config = cfg.dump()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    prepare_run(cfg, training=False)
    train_loader, val_loaders, _, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_classes).cuda()
    loss_fn = make_loss(cfg, num_classes)
    optimizer = make_optimizer(cfg, model)
    # 只为一次性数值预检使用较低初始缩放，避免默认高缩放的初期 AMP
    # 跳步被误判为模型梯度失效；正式训练仍沿用原默认 GradScaler。
    scaler = amp.GradScaler(init_scale=1024.0)
    imgs, pids, cams = next(iter(train_loader))
    assert imgs[0].shape[0] == cfg.SOLVER.IMS_PER_BATCH
    imgs = [img.cuda() for img in imgs]
    pids = pids.cuda()
    cams = [cam.cuda() for cam in cams]
    torch.cuda.reset_peak_memory_stats()
    model.train()
    with amp.autocast():
        score, feature, _ = model(imgs, pids, cams)
        loss, ce, tri = loss_fn(score, feature, pids.repeat(len(imgs)))
    assert torch.isfinite(loss) and torch.isfinite(feature).all()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), '梯度非有限值'
    gain_gradient = None
    if cfg.MODEL.TOKEN_TRAJECTORY:
        g = model.base.token_trajectory.gain.grad
        assert g is not None and g.abs().sum() > 0, 'Trajectory 未收到非零梯度'
        gain_gradient = float(g.abs().max())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    del score, feature, loss, imgs, grads
    train_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    # 使用真实 TEST.IMS_PER_BATCH，检查 fp32 推理显存及两种读出。
    image = next(iter(val_loaders[0]))[0].cuda()
    assert image.shape[0] == cfg.TEST.IMS_PER_BATCH
    model.eval()
    with torch.no_grad():
        model.neck_feat = 'before'
        before = model(image, mode=1)
        model.neck_feat = 'after'
        after = model(image, mode=1)
    assert before.shape == after.shape == (cfg.TEST.IMS_PER_BATCH, 768)
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    import yaml
    write_json(Path(args.output_dir) / 'preflight_ok.json', {
        'passed': True, 'experiment_id': cfg.EXPERIMENT.ID,
        'config': yaml.safe_load(original_config), 'runtime': runtime_info(),
        'logical_batch': cfg.SOLVER.IMS_PER_BATCH,
        'actual_images': cfg.SOLVER.IMS_PER_BATCH * len(cfg.DATASETS.MODALITIES),
        'eval_batch': cfg.TEST.IMS_PER_BATCH, 'ce': float(ce), 'triplet': float(tri),
        'preflight_amp_initial_scale': 1024.0,
        'gain_gradient_max': gain_gradient,
        'train_peak_gib': train_peak / 1024**3,
        'eval_peak_gib': torch.cuda.max_memory_allocated() / 1024**3})
    print('PREFLIGHT_OK：仅数值预检通过，正式 60 epoch 训练尚未启动。')


if __name__ == '__main__':
    main()
