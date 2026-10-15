#!/usr/bin/env bash
# P3：单张 A800 前台训练完整 60 epoch；结束后用同一权重测试两种读出。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory_history_p3
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory_history_p3/configs/history_P3.yml
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_runs/history_round2_1016/P3_seed1234
if [[ -e "$RUN_DIR" ]]; then echo "输出目录已存在，请给本次启动使用独立目录：$RUN_DIR" >&2; exit 1; fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
cp "$CONFIG" "$RUN_DIR/selected_config.yml"
COMMON_OPTS=(
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets
)
# 本仓库的完整训练入口实际为 train.py，没有 finetune.py。
"$PYTHON_BIN" -u /mnt/cache/wanghanzhi/XK/Trajectory_history_p3/train.py \
  --config_file "$CONFIG" "${COMMON_OPTS[@]}" HISTORY.STOP_AFTER_EPOCH 0 OUTPUT_DIR "$RUN_DIR"
"$PYTHON_BIN" -u /mnt/cache/wanghanzhi/XK/Trajectory_history_p3/test.py \
  --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
"$PYTHON_BIN" -u /mnt/cache/wanghanzhi/XK/Trajectory_history_p3/test.py \
  --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"
