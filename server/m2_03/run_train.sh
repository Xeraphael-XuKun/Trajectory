#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory-m2-c
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/Trajectory-m2-c/train.py \
  --config_file /mnt/cache/wanghanzhi/XK/Trajectory-m2-c/configs/m2_03_deep_text_c0_control.yml
