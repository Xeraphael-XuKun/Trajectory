#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory
# 训练成功退出后，再自动执行两次独立测试；失败立即停止。
bash /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/server/hi3_m2e/run_train.sh "$@"
bash /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/server/hi3_m2e/eval_epoch60.sh
