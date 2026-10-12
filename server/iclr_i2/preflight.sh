#!/usr/bin/env bash
# 正式服务器资源与192图像检查；要求已生成完整共享缓存，禁止本地插值兼容选项。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/neutral_preflight.py \
  --resources "$ROOT/configs/iclr_exp/i2_resources_server.yml" \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I2_preflight \
  --device cuda --batch-per-modality 64 --require-stats
