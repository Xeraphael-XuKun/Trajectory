#!/usr/bin/env bash
# HD1：单卡前台运行前5epoch，保留60epoch学习率日程。末尾保存诊断权重，不自动检索评估。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory_history
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory_history/configs/history_HD1.yml
# 如需重新从初始化检查前10epoch，显式传入参数10；不是断点续训。
STOP_EPOCH=${1:-5}
case "$STOP_EPOCH" in 5|10) ;; *) echo "诊断停止epoch仅支持5或10" >&2; exit 1 ;; esac
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_runs/history_optimization_1009/HD1_e${STOP_EPOCH}_seed1234
if [[ -e "$RUN_DIR" ]]; then echo "输出已存在，请修改RUN_DIR为独立retry目录：$RUN_DIR" >&2; exit 1; fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
"$PYTHON_BIN" -u train.py --config_file "$CONFIG" \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  OUTPUT_DIR "$RUN_DIR" HISTORY.STOP_AFTER_EPOCH "$STOP_EPOCH"
echo "HD1诊断完成：$RUN_DIR。仅完成${STOP_EPOCH}epoch，不能当作60epoch性能结果。"
