#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 train.py --config_file configs/m2_04_clip_relation_distill.yml
