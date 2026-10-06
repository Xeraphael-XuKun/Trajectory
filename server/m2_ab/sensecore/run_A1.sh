#!/usr/bin/env bash
# AB-A1：单卡前台训练 60 epoch，然后从同一落盘权重分别测试 post/pre-BN。
# 正式训练入口沿用本仓库 train.py；预检单独执行，不作为强制门槛。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1
CODE_DIR="/mnt/cache/wanghanzhi/XK/Trajectory_m2"
PYTHON="/mnt/cache/wanghanzhi/envs/llmpar/bin/python3"
CONFIG="$CODE_DIR/configs/m2_ab/AB-A1.yml"
RUN_DIR="${M2_RUN_DIR:-/mnt/cache/wanghanzhi/XK/Trajectory_runs/m2_ab_1007/AB-A1_seed1234}"
if [[ -e "$RUN_DIR" ]]; then
  echo "输出目录已存在，请先核实；另起实验可明确设置 M2_RUN_DIR，避免覆盖：$RUN_DIR" >&2
  exit 1
fi
cd "$CODE_DIR"
mkdir -p "$RUN_DIR"
# 保存本次源代码版本和是否有未提交改动，便于解释实验来源。
if git rev-parse HEAD > "$RUN_DIR/source_commit.txt" 2>/dev/null; then
  git status --short > "$RUN_DIR/source_status.txt"
else
  echo "当前服务器目录无 Git 元数据；请在实验记录中注明部署版本。" > "$RUN_DIR/source_commit.txt"
fi
"$PYTHON" -u "$CODE_DIR/train.py" --config_file "$CONFIG" OUTPUT_DIR "$RUN_DIR"
CHECKPOINT="$RUN_DIR/transformer_60.pth"
"$PYTHON" -u "$CODE_DIR/test.py" --config_file "$CONFIG" TEST.WEIGHT "$CHECKPOINT" TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
"$PYTHON" -u "$CODE_DIR/test.py" --config_file "$CONFIG" TEST.WEIGHT "$CHECKPOINT" TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"
