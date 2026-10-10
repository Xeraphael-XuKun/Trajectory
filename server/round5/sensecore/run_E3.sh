#!/usr/bin/env bash
# E3 单卡前台完整训练，再重新加载同一 epoch60 权重测试 post/pre。
# 预检单独运行；该脚本不运行 smoke、不安装环境、不覆盖既有实验。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_E3
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_runs/trajectory_round5_1011/E3_seed1234
CONFIG="$CODE_DIR/configs/trajectory_E3_uniform_transport.yml"
cd "$CODE_DIR"
[[ -x "$PYTHON_BIN" && -f "$CLIP_WEIGHT" && -f "$CONFIG" ]]
for split in train query test; do
  [[ -d "$DATA_ROOT/WHU-MARS/$split" ]]
done
if [[ -e "$RUN_DIR" ]]; then
  echo "输出目录已存在，停止以免覆盖：$RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
{
  echo '基准提交：48b6ba97a56292b3eb71374d81c713e742eb2075'
  echo '实际提交：'
  git rev-parse HEAD
  echo '工作区差异：'
  git status --short
  git diff --stat
  echo '解释器：/mnt/cache/wanghanzhi/envs/llmpar/bin/python3'
  echo '单卡：CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1'
  "$PYTHON_BIN" -c 'from utils.runtime import runtime_summary; print(runtime_summary())'
} > "$RUN_DIR/manifest.txt"
COMMON_OPTS=(MODEL.PRETRAIN_PATH "$CLIP_WEIGHT" DATASETS.ROOT_DIR "$DATA_ROOT")
COMMAND=("$PYTHON_BIN" -u "$CODE_DIR/train.py" --config_file "$CONFIG"
  "${COMMON_OPTS[@]}" SOLVER.MAX_EPOCHS 60 SOLVER.SEED 1234
  SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.00035 SOLVER.PRETRAINED_LR 0.000005
  MODEL.TOKEN_TRAJECTORY_VARIANT velocity_uniform_transport MODEL.TOKEN_TRAJECTORY_ACCEL_MIX 0.0
  TEST.NECK_FEAT after OUTPUT_DIR "$RUN_DIR")
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/console.log"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT after \
  OUTPUT_DIR "$RUN_DIR/eval_postBN" 2>&1 | tee "$RUN_DIR/test_post_console.log"
"$PYTHON_BIN" -u "$CODE_DIR/test.py" --config_file "$CONFIG" "${COMMON_OPTS[@]}" \
  TEST.WEIGHT "$RUN_DIR/transformer_60.pth" TEST.NECK_FEAT before \
  OUTPUT_DIR "$RUN_DIR/eval_preBN" 2>&1 | tee "$RUN_DIR/test_pre_console.log"
