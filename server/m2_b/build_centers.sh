#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-b/Trajectory
# 仅提取 train/RGB；其他光谱仅扫描文件名以建立完整 PID 映射与条件掩码。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/tools/m2/build_clip_cache.py \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output /mnt/cache/wanghanzhi/XK/m2-b/artifacts/train_clip_features.pt
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/tools/m2/build_rgb_center_cache.py \
  --input /mnt/cache/wanghanzhi/XK/m2-b/artifacts/train_clip_features.pt \
  --output /mnt/cache/wanghanzhi/XK/m2-b/artifacts/rgb_centers.pt
