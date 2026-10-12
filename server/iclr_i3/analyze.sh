#!/usr/bin/env bash
# 在相同索引/协议下比较I3与旧M5/C0，保存改善、削弱和不变query。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/analyze.py \
  --evaluation /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I3_M5_Paired_seed1234/evaluation \
  --i2-evaluation /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I2_M5_Neutral_seed1234/evaluation \
  --m5-evaluation /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_references/M5 \
  --c0-evaluation /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_references/C0 \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I3_M5_Paired_seed1234/handoff
