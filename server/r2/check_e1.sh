#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory
# 独立六图预检；退出后再启动正式训练，不复用预检模型或随机状态。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/tests/r2/test_kd_algebra.py
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/scripts/r2/check_e1.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/configs/r2_m5_flat_tail.yml \
  --output /mnt/cache/wanghanzhi/XK/m2-r2-e1/results/preflight
