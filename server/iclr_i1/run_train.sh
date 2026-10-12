#!/usr/bin/env bash
# 前台执行 I1 完整60epoch学生训练，不自动训练教师或身份prompt。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/train.py \
  --base-config "$ROOT/configs/trajectory_C0.yml" \
  --common-config "$ROOT/configs/iclr_exp/common.yml" \
  --config "$ROOT/configs/iclr_exp/tgva_ref.yml" \
  --resources "$ROOT/configs/iclr_exp/resources_server.yml" "$@"
