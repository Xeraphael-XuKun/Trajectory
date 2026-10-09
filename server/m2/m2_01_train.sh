#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u train.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/configs/m2_01_condition_text_bridge.yml \
  --local_rank 0
