#!/usr/bin/env bash
# L_S_B：llmpar，symmetric，Baseline；60 epoch 后独立测试 pre-BN / post-BN。
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/Trajectory
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
mkdir -p /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001
# 直接启动正式训练；预检与配置检查均为可选操作。

/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/Trajectory/finetune.py --config_file /mnt/cache/wanghanzhi/XK/Trajectory/configs/L_S_B.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  MODEL.TOKEN_TRAJECTORY False MODEL.TEXT_ALIGN False \
  INPUT.PIXEL_MEAN '[0.5, 0.5, 0.5]' INPUT.PIXEL_STD '[0.5, 0.5, 0.5]' \
  SOLVER.MAX_EPOCHS 60 SOLVER.SEED 1234 \
  SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 DATALOADER.NUM_WORKERS 8 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 \
  SOLVER.WEIGHT_DECAY 0.0001 SOLVER.WEIGHT_DECAY_BIAS 0.0001 \
  SOLVER.WARMUP_ITERS 100 SOLVER.PE_DELTA_LR_MULT 1.0 \
  SOLVER.CUDNN_BENCHMARK True SOLVER.CUDNN_DETERMINISTIC True \
  SOLVER.TEXT_LOSS_WEIGHT 0.0 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.EVAL_PERIOD 10 \
  TEST.IMS_PER_BATCH 1024 TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes TEST.METRIC sysu TEST.RE_RANKING False TEST.TOP_K_EVAL 0 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B_seed1234 2>&1 | tee -a /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B.train_console.txt

/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/Trajectory/test.py --config_file /mnt/cache/wanghanzhi/XK/Trajectory/configs/L_S_B.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  MODEL.TOKEN_TRAJECTORY False MODEL.TEXT_ALIGN False \
  INPUT.PIXEL_MEAN '[0.5, 0.5, 0.5]' INPUT.PIXEL_STD '[0.5, 0.5, 0.5]' \
  SOLVER.MAX_EPOCHS 60 SOLVER.SEED 1234 \
  SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 DATALOADER.NUM_WORKERS 8 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 \
  SOLVER.WEIGHT_DECAY 0.0001 SOLVER.WEIGHT_DECAY_BIAS 0.0001 \
  SOLVER.WARMUP_ITERS 100 SOLVER.PE_DELTA_LR_MULT 1.0 \
  SOLVER.CUDNN_BENCHMARK True SOLVER.CUDNN_DETERMINISTIC True \
  SOLVER.TEXT_LOSS_WEIGHT 0.0 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.EVAL_PERIOD 10 \
  TEST.IMS_PER_BATCH 1024 TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes TEST.METRIC sysu TEST.RE_RANKING False TEST.TOP_K_EVAL 0 \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B_seed1234/transformer_60.pth \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B_seed1234/pre_bn 2>&1 | tee -a /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B.pre_console.txt

/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/Trajectory/test.py --config_file /mnt/cache/wanghanzhi/XK/Trajectory/configs/L_S_B.yml \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  MODEL.TOKEN_TRAJECTORY False MODEL.TEXT_ALIGN False \
  INPUT.PIXEL_MEAN '[0.5, 0.5, 0.5]' INPUT.PIXEL_STD '[0.5, 0.5, 0.5]' \
  SOLVER.MAX_EPOCHS 60 SOLVER.SEED 1234 \
  SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 DATALOADER.NUM_WORKERS 8 \
  SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005 \
  SOLVER.WEIGHT_DECAY 0.0001 SOLVER.WEIGHT_DECAY_BIAS 0.0001 \
  SOLVER.WARMUP_ITERS 100 SOLVER.PE_DELTA_LR_MULT 1.0 \
  SOLVER.CUDNN_BENCHMARK True SOLVER.CUDNN_DETERMINISTIC True \
  SOLVER.TEXT_LOSS_WEIGHT 0.0 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.EVAL_PERIOD 10 \
  TEST.IMS_PER_BATCH 1024 TEST.NECK_FEAT after \
  TEST.FEAT_NORM yes TEST.METRIC sysu TEST.RE_RANKING False TEST.TOP_K_EVAL 0 \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B_seed1234/transformer_60.pth \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B_seed1234/post_bn 2>&1 | tee -a /mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001/L_S_B.post_console.txt
