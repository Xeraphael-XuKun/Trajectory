#!/usr/bin/env bash
# 旧M5和C0仅重评测，不重训；参考特征可供三个新组复用。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/eval_and_export.py \
  --resources "$ROOT/configs/iclr_exp/i2_resources_server.yml" \
  --reference-method m5 \
  --checkpoint /mnt/cache/wanghanzhi/XK/m2-e/logs/student/transformer_60_student.pth \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_references/M5 \
  --device cuda --batch-size 1024 --workers 8
exec /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/iclr_exp/eval_and_export.py \
  --resources "$ROOT/configs/iclr_exp/i2_resources_server.yml" \
  --reference-method c0 \
  --reference-log /mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_extensions_1002/C0_seed1234/train_log_20261001_2359.txt \
  --checkpoint /mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_extensions_1002/C0_seed1234/transformer_60.pth \
  --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/iclr_references/C0 \
  --device cuda --batch-size 1024 --workers 8
