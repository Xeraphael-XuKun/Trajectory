#!/usr/bin/env bash
# 单组预检，和正式训练严格分开。例：bash server/preflight.sh W_S_B
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/Trajectory
case "${1:-}" in
  W_S_B|W_S_T|W_C_B|W_C_T)
    PYTHON_BIN=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 ;;
  L_S_B|L_S_T|L_C_B|L_C_T)
    PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 ;;
  *) echo "用法：bash server/preflight.sh {W_S_B|W_S_T|W_C_B|W_C_T|L_S_B|L_S_T|L_C_B|L_C_T}" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
if [ -e "/mnt/cache/wanghanzhi/XK/Trajectory/logs/preflight_1001/$1" ] || [ -e "/mnt/cache/wanghanzhi/XK/Trajectory/logs/preflight_1001/$1.console.txt" ]; then
  echo "本组预检输出已存在，请保留失败证据并使用独立重试目录。" >&2
  exit 1
fi
mkdir -p /mnt/cache/wanghanzhi/XK/Trajectory/logs/preflight_1001
"$PYTHON_BIN" -u /mnt/cache/wanghanzhi/XK/Trajectory/tools/preflight.py \
  --config_file "/mnt/cache/wanghanzhi/XK/Trajectory/configs/$1.yml" \
  --output_dir "/mnt/cache/wanghanzhi/XK/Trajectory/logs/preflight_1001/$1" \
  2>&1 | tee "/mnt/cache/wanghanzhi/XK/Trajectory/logs/preflight_1001/$1.console.txt"
