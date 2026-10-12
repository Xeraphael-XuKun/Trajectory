#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e2/Trajectory
# 前台仅训练E2学生60epoch；教师/prompt仅复用，不调用原全阶段管线。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e2/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-r2-e2/Trajectory/configs/r2_m5_one_sided.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt M2.CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS \
  M2.PROMPT_WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/artifacts/shared_id_prompts_final.pt \
  M2.TEXT_BANK /mnt/cache/wanghanzhi/XK/m2-e/artifacts/shared_id_text_bank.pt \
  M2.TEACHER_WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/logs/teacher/conditional_teacher_epoch60.pth \
  SOLVER.SEED 1234 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 SOLVER.WEIGHT_DECAY 0.0001 \
  M2.TEXT_TEMPERATURE 0.07 M2.KD_TEMPERATURE 2.0 M2.KD_WEIGHT 0.5 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-r2-e2/results/E2
