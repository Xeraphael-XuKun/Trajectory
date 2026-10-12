#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory
OUTPUT=/mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/results/W0
mkdir -p "$OUTPUT"
printf '%s\n' -1 > "$OUTPUT/launcher_exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT/launcher_exit_code.txt"' EXIT
# 前台核验原工件和严格加载；不进行任何优化或训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u scripts/r2/audit_artifacts.py \
  --resources /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory/server/r2/resources_server.json \
  --output "$OUTPUT" --device cpu --formal 2>&1 | tee "$OUTPUT/run.log"
