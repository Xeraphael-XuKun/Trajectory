#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -m py_compile \
  model/m2/condition_bridge.py tools/m2/build_train_clip_cache.py \
  tools/m2/train_condition_prompt.py model/make_model.py \
  processor/processor.py train.py

