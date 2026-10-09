#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory-m2-c
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u server/m2_03/build_feature_cache.py \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output /mnt/cache/wanghanzhi/XK/Trajectory-m2-c/cache/m2_03_train_features.pt
