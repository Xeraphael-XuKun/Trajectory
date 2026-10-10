#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-c/Trajectory
# 阶段 A 工件完成后，只运行必要小批量方法检查。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/tools/m2/check_m2_3.py \
  --config /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/configs/m2_03_deep_text_c0_control.yml --device cuda
