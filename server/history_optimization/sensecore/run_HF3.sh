#!/usr/bin/env bash
# HF3：诊断后显式传入HD0/HD1/HD2/HD3，并先生成配置；从初始化完整训练60epoch。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory_history
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
CHOICE=${1:?请显式提供已选优化设置HD0、HD1、HD2或HD3}
case "$CHOICE" in HD0|HD1|HD2|HD3) ;; *) echo "无效优化设置：$CHOICE" >&2; exit 1 ;; esac
CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory_history/configs/history_followup/$CHOICE/history_HF3.yml
if [[ ! -f "$CONFIG" ]]; then echo "请先用prepare_history_followup.py --choice $CHOICE生成配置" >&2; exit 1; fi
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_runs/history_followup_1009/${CHOICE}_HF3_seed1234
if [[ -e "$RUN_DIR" ]]; then echo "输出已存在，请使用独立retry目录：$RUN_DIR" >&2; exit 1; fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
cp "$CONFIG" "$RUN_DIR/selected_config.yml"
COMMON_OPTS=(MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets)
"$PYTHON_BIN" -u train.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" OUTPUT_DIR "$RUN_DIR"
"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR/eval_postBN"
"$PYTHON_BIN" -u test.py --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT before OUTPUT_DIR "$RUN_DIR/eval_preBN"
