#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-d/Trajectory
# 同一epoch60纯学生；关闭M2，不读取教师或中心工件。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/configs/m2_04_clip_relation_distill.yml \
  M2.ENABLED False MODEL.PRETRAIN_CHOICE no \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-d/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-d/logs/eval_epoch60_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/configs/m2_04_clip_relation_distill.yml \
  M2.ENABLED False MODEL.PRETRAIN_CHOICE no \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-d/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-d/logs/eval_epoch60_preBN
