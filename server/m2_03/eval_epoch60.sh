#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-c/Trajectory
# 保留 M2-3 视觉控制器；部署入口不构造文本塔，也不读 prompt 工件。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/configs/m2_03_deep_text_c0_control.yml \
  MODEL.PRETRAIN_CHOICE no M2.DEPLOY_ONLY True \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-c/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-c/logs/eval_epoch60_postBN
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/test.py \
  --config_file /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/configs/m2_03_deep_text_c0_control.yml \
  MODEL.PRETRAIN_CHOICE no M2.DEPLOY_ONLY True \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-c/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-c/logs/eval_epoch60_preBN
