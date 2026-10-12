#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory
OUTPUT=/mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/results/D2
mkdir -p "$OUTPUT"
printf '%s\n' -1 > "$OUTPUT/launcher_exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT/launcher_exit_code.txt"' EXIT
# 原epoch60、原测试batch1024、原排序块4000；同一次视觉前向读取pre/post。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u scripts/r2/extract_and_pair.py \
  --resources /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory/server/r2/resources_server.json \
  --output "$OUTPUT" --device cuda --batch-size 1024 --workers 8 --rank-chunk 4000 --formal \
  2>&1 | tee "$OUTPUT/run.log"
