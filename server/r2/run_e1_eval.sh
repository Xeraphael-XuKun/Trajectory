#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory
# 完整训练结束后，独立重载同一epoch60，post主读出、pre补充。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/configs/r2_m5_flat_tail.yml \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-r2-e1/results/E1/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-r2-e1/results/E1/eval_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-r2-e1/Trajectory/configs/r2_m5_flat_tail.yml \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-r2-e1/results/E1/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-r2-e1/results/E1/eval_preBN
