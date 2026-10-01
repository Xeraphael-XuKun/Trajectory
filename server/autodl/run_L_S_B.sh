#!/usr/bin/env bash
# AutoDL A800：L_S_B，训练 60 epoch 后补测同一权重的 pre-BN / post-BN。
# 只适配路径；复用原八组 YAML、train.py、test.py，保留日志格式。
# 默认路径沿用已有 AutoDL 目录约定，需与本实例实际路径一致。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=${TRAJECTORY_CODE_DIR:-/root/autodl-tmp/XK/Trajectory}
PYTHON_BIN=${LLMPAR_PYTHON:-/root/autodl-tmp/envs/llmpar/bin/python3}
DATA_ROOT=${TRAJECTORY_DATA_ROOT:-/root/autodl-tmp}
CLIP_WEIGHT=${TRAJECTORY_CLIP_WEIGHT:-/root/autodl-tmp/ViT-B-16.pt}
RUN_DIR=${TRAJECTORY_RUN_DIR:-/root/autodl-tmp/XK/Trajectory_runs/autodl/matrix_minimal_1001/L_S_B_seed1234}
CONFIG="$CODE_DIR/configs/L_S_B.yml"
cd "$CODE_DIR"

COMMON_OPTS=(
  MODEL.PRETRAIN_PATH "$CLIP_WEIGHT"
  MODEL.TEXT_CLIP_PATH "$CLIP_WEIGHT"
  DATASETS.ROOT_DIR "$DATA_ROOT"
)

"$PYTHON_BIN" -u train.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  OUTPUT_DIR "$RUN_DIR"

"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"

"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
