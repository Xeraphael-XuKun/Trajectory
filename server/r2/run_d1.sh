#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory
OUTPUT=/mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/results/D1
mkdir -p "$OUTPUT"
printf '%s\n' -1 > "$OUTPUT/launcher_exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT/launcher_exit_code.txt"' EXIT
# 全训练身份确定性诊断及有限增强eval诊断；教师学生使用同一张量。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u scripts/r2/analyze_teacher.py \
  --resources /mnt/cache/wanghanzhi/XK/m2-r2-diagnostics/Trajectory/server/r2/resources_server.json \
  --output "$OUTPUT" --device cuda --batch-size 64 --workers 8 --formal \
  2>&1 | tee "$OUTPUT/run.log"
