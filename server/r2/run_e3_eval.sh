#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory
# 同一epoch60重载，保留controller/context与C0；移除训练文本塔。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/configs/r2_m3_no_wd.yml \
  MODEL.PRETRAIN_CHOICE no M2.DEPLOY_ONLY True \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3/eval_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/configs/r2_m3_no_wd.yml \
  MODEL.PRETRAIN_CHOICE no M2.DEPLOY_ONLY True \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3/eval_preBN
