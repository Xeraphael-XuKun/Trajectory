#!/usr/bin/env bash
# HI0：单卡前台训练 60 epoch，随后重新加载同一权重测试 post-BN 和 pre-BN。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory_history
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory_history/configs/history_HI0.yml
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_runs/history_innovation_1009/HI0_seed1234
# 防止误覆盖已完成或中断的实验。重试时请将 RUN_DIR 改为新的 retry 后缀目录。
if [[ -e "$RUN_DIR" ]]; then echo "输出目录已存在，请使用新的 RUN_DIR：$RUN_DIR" >&2; exit 1; fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
COMMON_OPTS=(
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets
)
"$PYTHON_BIN" -u train.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" OUTPUT_DIR "$RUN_DIR"
"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"
