#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
# 仅生成 train 原始冻结教师特征。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/tools/m2/build_clip_cache.py \
  --data-root /mnt/cache/wanghanzhi/Datasets --subdir WHU-MARS \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output /mnt/cache/wanghanzhi/XK/m2-a/artifacts/train_clip_features.pt
