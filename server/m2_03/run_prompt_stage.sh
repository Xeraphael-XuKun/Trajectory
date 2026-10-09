#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 server/m2_03/prompt_stage.py \
  --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --feature-cache /mnt/cache/wanghanzhi/XK/Trajectory-m2-c/cache/m2_03_train_features.pt \
  --output /mnt/cache/wanghanzhi/XK/Trajectory-m2-c/cache/m2_03_prompt_bank.pt \
  --steps 10000
