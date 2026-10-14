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
# 与预检分开，前台执行原train.py的完整60epoch训练。
mkdir -p "$OUTPUT"
"$PYTHON" -u "$REPO/train.py" --config_file "$REPO/configs/m2_dt/direct_text_c0.yml" "$@" 2>&1 | tee -a "$OUTPUT/train.log"
