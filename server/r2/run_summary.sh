#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory
# 只汇总已有完成manifest；不将残缺特征/中途指标当作完成结果。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u scripts/r2/summarize_diagnostics.py \
  --root /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/results
