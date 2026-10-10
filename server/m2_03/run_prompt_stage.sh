#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-c/Trajectory
# 读取 PROMPT_STAGE 全部设置，固定 seed1234 和10000次真实更新，前台执行。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/server/m2_03/prompt_stage.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/configs/m2_03_deep_text_c0_control.yml
