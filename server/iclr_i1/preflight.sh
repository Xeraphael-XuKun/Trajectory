#!/usr/bin/env bash
# 必要小样本预检；与正式完整训练分开。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/preflight.py \
  --resources "$ROOT/configs/iclr_exp/resources_server.yml" --device cuda \
  --batch-per-modality 64 \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I1_preflight
