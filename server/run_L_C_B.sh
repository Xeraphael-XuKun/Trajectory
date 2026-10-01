#!/usr/bin/env bash
# L_C_B：训练 60 epoch，随后用同一权重测试 pre-BN 与 post-BN。
# 保留 0158be2 的 train.py/test.py 及原始日志格式；无强制预检。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory

PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory/configs/L_C_B.yml
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_minimal_1001/L_C_B_seed1234
COMMON_OPTS=(
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
  MODEL.TEXT_CLIP_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets
)

"$PYTHON_BIN" -u train.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  OUTPUT_DIR "$RUN_DIR"

"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"

"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
