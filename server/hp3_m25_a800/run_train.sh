#!/usr/bin/env bash
# 单卡前台完整训练；原逻辑batch64、60epoch、原A800教师，拒绝覆盖已有输出。
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 OMP_NUM_THREADS=8
export PYTHONUTF8=1
PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
ROOT=/mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018
OUT=/mnt/cache/wanghanzhi/XK/Trajectory_runs/hp3_m25_a800_1018/HP3_M25_seed1234
if [[ -e "$OUT" ]]; then echo "保留已有正式输出，请先核对：$OUT" >&2; exit 1; fi
"$PYTHON" -c 'import torch; name=torch.cuda.get_device_name(0); print(name); assert "A800" in name, "本实验要求单卡A800"'
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
trap 'code=$?; printf "FAILED exit=%s\n" "$code" >> "$OUT/status.txt"; exit "$code"' ERR
cp "$ROOT/configs/hp3_m25_a800.yml" "$OUT/selected_config.yml"
"$PYTHON" -c 'from tools.m2.artifacts import code_commit; print(code_commit())' > "$OUT/source_commit.txt"
printf 'TRAINING_STARTED\n' > "$OUT/status.txt"
"$PYTHON" -u "$ROOT/finetune.py" --config_file "$ROOT/configs/hp3_m25_a800.yml"
test -s "$OUT/transformer_60_student.pth"
printf 'TRAINING_COMPLETED\n' >> "$OUT/status.txt"
bash "$ROOT/server/hp3_m25_a800/eval_epoch60.sh"
printf 'COMPLETED_WITH_ALL_POST_PRE_EVALUATIONS\n' >> "$OUT/status.txt"
