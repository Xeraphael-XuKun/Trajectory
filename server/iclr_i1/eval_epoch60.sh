#!/usr/bin/env bash
# 独立进程严格重载，post-BN主评测和同权重pre-BN补充。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/eval_and_export.py \
  --resources "$ROOT/configs/iclr_exp/resources_server.yml" \
  --checkpoint /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I1_C0_TGVA_ref_seed1234/transformer_60_student.pth \
  --full-checkpoint /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I1_C0_TGVA_ref_seed1234/transformer_60.pth \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I1_C0_TGVA_ref_seed1234/eval_epoch60
