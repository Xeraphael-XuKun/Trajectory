#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory-m2-e
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 train.py --config_file configs/m2_05_conditional_teacher_distill.yml
