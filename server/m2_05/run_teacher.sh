#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 前台条件教师60epoch；原PKM/增强/CE/Triplet/调度器。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/tools/m2/train_conditional_teacher.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt M2.CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS \
  M2.PROMPT_WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/artifacts/shared_id_prompts_final.pt \
  M2.TEXT_BANK /mnt/cache/wanghanzhi/XK/m2-e/artifacts/shared_id_text_bank.pt \
  SOLVER.SEED 1234 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 SOLVER.WEIGHT_DECAY 0.0001 \
  TEACHER_STAGE.MAX_EPOCHS 60 TEACHER_STAGE.LORA_RANK 8 TEACHER_STAGE.LORA_ALPHA 8.0 \
  TEACHER_STAGE.TEMPERATURE 0.07 TEACHER_STAGE.TEXT_ID_WEIGHT 0.5 \
  TEACHER_STAGE.OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-e/logs/teacher
