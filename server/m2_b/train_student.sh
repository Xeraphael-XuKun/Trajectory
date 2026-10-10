#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-b/Trajectory
# 正式训练前台执行，与预检和缓存生成分开。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/configs/m2_2.yml
