#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
# 两次评测使用同一 epoch60 学生；原C0部署结构，不读取训练文本工件。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/configs/m2_01_condition_text_bridge.yml \
  M2.ENABLED False MODEL.PRETRAIN_CHOICE no \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-a/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-a/logs/eval_epoch60_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/configs/m2_01_condition_text_bridge.yml \
  M2.ENABLED False MODEL.PRETRAIN_CHOICE no \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-a/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-a/logs/eval_epoch60_preBN
