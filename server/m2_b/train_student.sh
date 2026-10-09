#!/usr/bin/env bash
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/m2-b/Trajectory
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 train.py --config_file configs/m2_2.yml
