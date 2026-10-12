#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory
# 新主模型60epoch；复用原初始化bank，不重跑prompt，不从旧学生续训。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/configs/r2_m3_no_wd.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt M2.CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS \
  M2.TEXT_BANK /mnt/cache/wanghanzhi/XK/m2-c/artifacts/deep_slot_prompt_bank.pt \
  M2.ID_PROMPT_WEIGHT /mnt/cache/wanghanzhi/XK/m2-c/artifacts/deep_slot_prompt_bank.pt \
  M2.DEEP_CONTEXT_INIT /mnt/cache/wanghanzhi/XK/m2-c/artifacts/deep_slot_prompt_bank.pt \
  SOLVER.SEED 1234 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 SOLVER.WEIGHT_DECAY 0.0001 \
  M2.CONTROL_RHO 0.25 M2.TEMPERATURE 0.07 M2.TEXT_ID_WEIGHT 0.5 M2.TEXT_ANCHOR_WEIGHT 0.1 \
  R2.CONTROLLER_WEIGHT_DECAY 0.0 OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3
