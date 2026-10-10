#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 10,000个实际prompt更新；从最后一步导出固定文本表。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/tools/m2/train_prompt_bank.py --config-file /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml --device cuda
