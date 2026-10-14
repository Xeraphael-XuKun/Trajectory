#!/usr/bin/env bash
set -euo pipefail
# 在仓库自身目录执行，克隆文件夹改名不会影响源码定位。
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
OUTPUT=/mnt/cache/wanghanzhi/XK/Trajectory_runs/m2_05_direct_text_c0_seed1234
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
[[ "$WORLD_SIZE" == 1 && "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "仅支持单卡 WORLD_SIZE=1"; exit 1; }
cd "$REPO"
# 两次读取同一部署checkpoint；严格加载全部模型state，不加载文本文件。
CHECKPOINT="$OUTPUT/transformer_60_student.pth"
for READOUT in after before; do
  if [[ "$READOUT" == after ]]; then NAME=eval_postBN; else NAME=eval_preBN; fi
  mkdir -p "$OUTPUT/$NAME"
  "$PYTHON" -u "$REPO/tools/m2_dt.py" --action eval --config "$REPO/configs/m2_dt/direct_text_c0.yml" TEST.WEIGHT "$CHECKPOINT" TEST.NECK_FEAT "$READOUT" OUTPUT_DIR "$OUTPUT/$NAME" 2>&1 | tee -a "$OUTPUT/$NAME/eval.log"
done
