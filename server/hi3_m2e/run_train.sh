#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory
# 从CLIP和零gain重新训练；复用原M2-e文本与教师，不加载HI3 epoch60。
if [[ -e /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/config_merged.yml && $# -eq 0 ]]; then
  echo '已有本组训练输出；请使用新输出目录或明确指定M2.RESUME。' >&2
  exit 1
fi
mkdir -p /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234
git rev-parse HEAD > /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/source_commit.txt
git status --short > /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/source_status.txt
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/finetune.py \
  --config_file /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/configs/hi3_m2e/hi3_original_m2e.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  M2.CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS \
  M2.PROMPT_WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/artifacts/shared_id_prompts_final.pt \
  M2.TEXT_BANK /mnt/cache/wanghanzhi/XK/m2-e/artifacts/shared_id_text_bank.pt \
  M2.TEACHER_WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/logs/teacher/conditional_teacher_epoch60.pth \
  SOLVER.SEED 1234 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 SOLVER.WEIGHT_DECAY 0.0001 \
  HISTORY.ALPHA 0.25 HISTORY.GAIN_MODE linear HISTORY.TOKEN_SCOPE patch \
  HISTORY.LOSS_WEIGHT 1.0 HISTORY.WRITE_WEIGHT_DECAY -1.0 \
  M2.TEXT_TEMPERATURE 0.07 M2.KD_TEMPERATURE 2.0 M2.KD_WEIGHT 0.5 \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234 "$@" 2>&1 | tee -a /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/train.log
