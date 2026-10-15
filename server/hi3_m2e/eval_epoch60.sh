#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory
# 同一个第60epoch纯学生：post主结果、pre补充；不构造教师或读取文本。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/test.py --config_file /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/configs/hi3_m2e/hi3_original_m2e.yml M2.ENABLED False MODEL.PRETRAIN_CHOICE no DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS TEST.WEIGHT /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/transformer_60_student.pth TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/eval_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/test.py --config_file /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/configs/hi3_m2e/hi3_original_m2e.yml M2.ENABLED False MODEL.PRETRAIN_CHOICE no DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets DATASETS.SUBDIR WHU-MARS TEST.WEIGHT /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/transformer_60_student.pth TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/eval_preBN
