#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 小样本方法检查，不代替正式教师或学生训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/tools/m2/check_method.py --config-file /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml --device cuda
