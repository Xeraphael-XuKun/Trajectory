#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-b/Trajectory
# 同一 epoch60 学生权重，post-BN 主结果、pre-BN 补充结果。测试不读取教师工件。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/configs/m2_2.yml \
  MODEL.PRETRAIN_CHOICE no M2.ENABLED False \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-b/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-b/logs/eval_epoch60_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-b/Trajectory/configs/m2_2.yml \
  MODEL.PRETRAIN_CHOICE no M2.ENABLED False \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-b/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-b/logs/eval_epoch60_preBN
