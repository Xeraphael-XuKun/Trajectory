#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-d/Trajectory
# 前台执行完整原训练链；原始CLIP开始，零gain。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/configs/m2_04_clip_relation_distill.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  M2.CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS \
  M2.RGB_CENTERS /mnt/cache/wanghanzhi/XK/m2-d/artifacts/rgb_identity_centers.pt \
  M2.RELATION_BANK /mnt/cache/wanghanzhi/XK/m2-d/artifacts/clip_identity_relation.pt \
  SOLVER.SEED 1234 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 SOLVER.WEIGHT_DECAY 0.0001 \
  M2.TEACHER_TEMPERATURE 0.07 M2.STUDENT_TEMPERATURE 0.07 M2.RELATION_WEIGHT 0.2 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-d/logs/student
