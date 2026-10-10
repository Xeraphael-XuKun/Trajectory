#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-b/Trajectory
# 教师中心已生成后再执行。只验证小批量方法，不启动正式训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/tools/m2/check_m2_2.py \
  --config /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/configs/m2_2.yml --device cuda
