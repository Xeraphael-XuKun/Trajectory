#!/usr/bin/env bash
# T3：EMA 历史外推，rho=0.25。
# 单卡前台训练60 epoch，再加载同一权重测试 pre-BN / post-BN。
# 当前项目的完整训练入口是 train.py；不安装环境，不自动恢复或覆盖旧输出。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=${TRAJECTORY_CODE_DIR:-/root/autodl-tmp/XK/Trajectory}
PYTHON_BIN=${LLMPAR_PYTHON:-/root/autodl-tmp/envs/llmpar/bin/python3}
DATA_ROOT=${TRAJECTORY_DATA_ROOT:-/root/autodl-tmp}
CLIP_WEIGHT=${TRAJECTORY_CLIP_WEIGHT:-/root/autodl-tmp/ViT-B-16.pt}
RUN_DIR=${TRAJECTORY_RUN_DIR:-/root/autodl-tmp/XK/Trajectory_runs/autodl/trajectory_extensions_1002/T3_seed1234}
CONFIG="$CODE_DIR/configs/trajectory_T3.yml"
cd "$CODE_DIR"
if [[ -e "$RUN_DIR" ]]; then
  echo "输出目录已存在：$RUN_DIR；请设置 TRAJECTORY_RUN_DIR 为新的独立目录。" >&2
  exit 1
fi
COMMON_OPTS=(
  MODEL.PRETRAIN_PATH "$CLIP_WEIGHT"
  MODEL.TEXT_CLIP_PATH "$CLIP_WEIGHT"
  DATASETS.ROOT_DIR "$DATA_ROOT"
)

"$PYTHON_BIN" -u "$CODE_DIR/train.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  OUTPUT_DIR "$RUN_DIR"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
