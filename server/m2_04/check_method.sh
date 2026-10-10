#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-d/Trajectory
# 必要方法小批量检查；不是正式学生训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/tools/check_m2_4.py \
  --config /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/configs/m2_04_clip_relation_distill.yml --device cuda
