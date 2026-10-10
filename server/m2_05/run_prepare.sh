#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 正式准备流程：缓存、prompt、教师。每阶段也有独立脚本。
bash /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/server/m2_05/run_cache.sh
bash /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/server/m2_05/run_prompt.sh
bash /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/server/m2_05/run_teacher.sh
