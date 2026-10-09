#!/bin/sh
set -eu
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/Trajectory/train.py --config_file /mnt/cache/wanghanzhi/XK/Trajectory/configs/m2_01_condition_text_bridge.yml --local_rank 0
