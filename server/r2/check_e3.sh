#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory
# 独立预检退出后，另起进程正式训练；不复用预检权重和随机状态。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/tests/r2/test_nowd_observations.py
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/scripts/r2/check_e3.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/configs/r2_m3_no_wd.yml \
  --output /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/preflight
