#!/usr/bin/env bash
# 冻结旧教师，只对真实训练集一次干净FP32统计；不启动训练。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/neutral_stats.py \
  --config "$ROOT/configs/iclr_exp/m5_neutral.yml" \
  --resources "$ROOT/configs/iclr_exp/i2_resources_server.yml" \
  --device cuda --batch-size 128 --workers 8 "$@"
