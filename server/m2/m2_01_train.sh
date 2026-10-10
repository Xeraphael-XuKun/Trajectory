#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
# 前台执行完整原训练链。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-a/Trajectory/configs/m2_01_condition_text_bridge.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  M2.CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS \
  M2.TEXT_BANK /mnt/cache/wanghanzhi/XK/m2-a/artifacts/condition_text_bank.pt \
  SOLVER.SEED 1234 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 SOLVER.WEIGHT_DECAY 0.0001 \
  M2.TEMPERATURE 0.07 M2.CROSS_TEMPERATURE 0.07 \
  M2.TEXT_ALL_WEIGHT 0.5 M2.TEXT_CROSS_WEIGHT 0.5 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-a/logs/student
