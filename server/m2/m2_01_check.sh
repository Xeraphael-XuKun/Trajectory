#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
# 正式工件准备好后的必要小批量检查；不是正式学生训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/tools/m2/check_m2_1.py \
  --config /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/configs/m2_01_condition_text_bridge.yml --device cuda
