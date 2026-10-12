#!/usr/bin/env bash
# 前台执行完整60epoch I3学生训练；不训练教师，不训练身份prompt。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/train.py \
  --base-config "$ROOT/configs/trajectory_C0.yml" \
  --common-config "$ROOT/configs/iclr_exp/common.yml" \
  --config "$ROOT/configs/iclr_exp/m5_paired.yml" \
  --resources "$ROOT/configs/iclr_exp/i3_resources_server.yml" "$@"
