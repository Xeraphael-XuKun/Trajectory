#!/usr/bin/env bash
# R4：纯速度方向与纯速度门控，门控无权重衰减。
# 单卡前台训练60 epoch，再加载同一权重测试 post-BN（主）/ pre-BN（补充）。
# 当前项目的完整训练入口是 train.py；不安装环境，不自动恢复或覆盖旧输出。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=${TRAJECTORY_CODE_DIR:-/mnt/cache/wanghanzhi/XK/Trajectory}
PYTHON_BIN=${LLMPAR_PYTHON:-/mnt/cache/wanghanzhi/envs/llmpar/bin/python3}
DATA_ROOT=${TRAJECTORY_DATA_ROOT:-/mnt/cache/wanghanzhi/Datasets}
CLIP_WEIGHT=${TRAJECTORY_CLIP_WEIGHT:-/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt}
RUN_DIR=${TRAJECTORY_RUN_DIR:-/mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_refinements_1002/R4_seed1234}
CONFIG="$CODE_DIR/configs/trajectory_R4.yml"
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
  TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" \
  TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"
