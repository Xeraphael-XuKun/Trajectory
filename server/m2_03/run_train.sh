#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-c/Trajectory
# 与预检/缓存/prompt 阶段分开；前台执行原训练链。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/configs/m2_03_deep_text_c0_control.yml
