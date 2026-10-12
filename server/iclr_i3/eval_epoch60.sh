#!/usr/bin/env bash
# 新进程严格重载完整/部署epoch60；保存pre/post及ALL/受限gallery逐query记录。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/eval_and_export.py \
  --config "$ROOT/configs/iclr_exp/m5_paired.yml" \
  --resources "$ROOT/configs/iclr_exp/i3_resources_server.yml" \
  --checkpoint /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I3_M5_Paired_seed1234/transformer_60_student.pth \
  --full-checkpoint /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I3_M5_Paired_seed1234/transformer_60.pth \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_I3_M5_Paired_seed1234/evaluation \
  --device cuda --batch-size 1024 --workers 8
