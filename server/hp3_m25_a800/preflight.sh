#!/usr/bin/env bash
# 独立预检：原教师严格加载、梯度与重载边界，再检查原192图生产训练循环。
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 OMP_NUM_THREADS=8
export PYTHONUTF8=1
PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
ROOT=/mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018
CHECK=/mnt/cache/wanghanzhi/XK/Trajectory_runs/hp3_m25_a800_1018/_preflight
if [[ -e "$CHECK" ]]; then echo "保留已有预检目录，请先核对：$CHECK" >&2; exit 1; fi
mkdir -p "$(dirname "$CHECK")"
mkdir "$CHECK"
exec > >(tee "$CHECK/console.log") 2>&1
"$PYTHON" -c 'import torch; name=torch.cuda.get_device_name(0); print(name); assert "A800" in name, "本实验要求单卡A800"'
"$PYTHON" -u "$ROOT/tools/hp3_m25/check_combination.py" \
  --config_file "$ROOT/configs/hp3_m25_a800.yml" --device cuda --output "$CHECK/boundaries"
"$PYTHON" -u "$ROOT/tools/preflight.py" \
  --config "$ROOT/configs/hp3_m25_a800.yml" --output "$CHECK/production"
printf 'A800_PREFLIGHT_COMPLETED\n' > "$CHECK/status.txt"
