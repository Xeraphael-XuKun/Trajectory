#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
# 条件均衡阶段 A；10000个实际更新与采样/LR预算由YAML读取。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/tools/m2/train_condition_prompt.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/configs/m2_01_condition_text_bridge.yml
