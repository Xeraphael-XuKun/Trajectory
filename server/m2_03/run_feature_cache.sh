#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-c/Trajectory
# 仅从 train 生成原始冻结 CLIP 的 RGB/IR/Thermal 特征。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/server/m2_03/build_feature_cache.py \
  --data-root /mnt/cache/wanghanzhi/Datasets --subdir WHU-MARS \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output /mnt/cache/wanghanzhi/XK/m2-c/artifacts/train_clip_features.pt
