#!/usr/bin/env bash
# I2/I3共用一次统计；存在时不重算、不覆盖，后续preflight核验其来源。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
if [[ -f /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_shared/neutral_view_shift.pt ]]; then
  echo "发现共享偏移缓存，跳过重复统计；请执行 I3 preflight 核验。"
  exit 0
fi
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/neutral_stats.py \
  --config "$ROOT/configs/iclr_exp/m5_paired.yml" \
  --resources "$ROOT/configs/iclr_exp/i3_resources_server.yml" \
  --device cuda --batch-size 128 --workers 8 "$@"
