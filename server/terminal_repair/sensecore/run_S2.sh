#!/usr/bin/env bash
# S2：终端排序修复，reference / joint / lambda=1.0。单卡前台完整训练，再从磁盘加载epoch60权重评估双读出。
# 真实训练入口为train.py；无需先执行预检，不覆盖已有输出。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
CODE_DIR=${TRAJECTORY_CODE_DIR:-/mnt/cache/wanghanzhi/XK/Trajectory}
PYTHON_BIN=${LLMPAR_PYTHON:-/mnt/cache/wanghanzhi/envs/llmpar/bin/python3}
DATA_ROOT=${TRAJECTORY_DATA_ROOT:-/mnt/cache/wanghanzhi/Datasets}
CLIP_WEIGHT=${TRAJECTORY_CLIP_WEIGHT:-/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt}
RUN_DIR=${TRAJECTORY_RUN_DIR:-/mnt/cache/wanghanzhi/XK/Trajectory_runs/terminal_repair_1008/S2_seed1234}
CONFIG="$CODE_DIR/configs/terminal_repair_S2.yml"
cd "$CODE_DIR"
if [[ -e "$RUN_DIR" ]]; then
  echo "输出目录已存在：$RUN_DIR；请设置TRAJECTORY_RUN_DIR为新的独立目录。" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
COMMON_OPTS=(MODEL.PRETRAIN_PATH "$CLIP_WEIGHT" MODEL.TEXT_CLIP_PATH "$CLIP_WEIGHT"
  DATASETS.ROOT_DIR "$DATA_ROOT")
"$PYTHON_BIN" -u "$CODE_DIR/train.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  SOLVER.MAX_EPOCHS 60 SOLVER.SEED 1234 SOLVER.CUDNN_BENCHMARK True TEST.NECK_FEAT after \
  OUTPUT_DIR "$RUN_DIR" 2>&1 | tee "$RUN_DIR/train_console.log"
test -s "$RUN_DIR/transformer_60.pth"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT after \
  OUTPUT_DIR "$RUN_DIR/eval_postBN" 2>&1 | tee "$RUN_DIR/postBN_console.log"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT before \
  OUTPUT_DIR "$RUN_DIR/eval_preBN" 2>&1 | tee "$RUN_DIR/preBN_console.log"
echo "TERMINAL_REPAIR_S2_TRAIN_AND_DISK_RELOAD_EVAL_OK"
