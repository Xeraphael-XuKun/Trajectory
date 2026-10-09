#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/cache/wanghanzhi/XK/m2-a/Trajectory
CACHE=/mnt/cache/wanghanzhi/XK/m2-a/artifacts/train_clip_cache.pt
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd "$PROJECT"
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/m2/build_clip_cache.py \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output "$CACHE"
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/m2/build_train_clip_cache.py --input "$CACHE" --output "$CACHE"
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/m2/train_condition_prompt.py \
  --cache "$CACHE" --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --output /mnt/cache/wanghanzhi/XK/m2-a/artifacts/condition_text_bank.pt --steps 10000
