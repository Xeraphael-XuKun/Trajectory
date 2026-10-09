#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/cache/wanghanzhi/XK/m2-b/Trajectory
CACHE=/mnt/cache/wanghanzhi/XK/m2-b/artifacts/train_clip_features.pt
OUT=/mnt/cache/wanghanzhi/XK/m2-b/artifacts/rgb_centers.pt
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd "$PROJECT"
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/m2/build_clip_cache.py \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output "$CACHE"
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/m2/build_rgb_center_cache.py --input "$CACHE" --output "$OUT"
