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
# 正式服务器预检：192张合成图的前后向、一次更新及严格重载，不训练/评测性能。
mkdir -p "$OUTPUT/preflight"
"$PYTHON" -u "$REPO/tools/m2_dt.py" --action preflight --config "$REPO/configs/m2_dt/direct_text_c0.yml" --device cuda --images-per-modality 64 --formal --output "$OUTPUT/preflight" 2>&1 | tee -a "$OUTPUT/preflight/check.log"
