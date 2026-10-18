#!/usr/bin/env bash
# 固定epoch60学生权重，ALL的post主/pre辅；推理不读取教师、文本或CLIP预训练文件。
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 OMP_NUM_THREADS=8
export PYTHONUTF8=1
PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
ROOT=/mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018
OUT=/mnt/cache/wanghanzhi/XK/Trajectory_runs/hp3_m25_a800_1018/HP3_M25_seed1234
test -s "$OUT/transformer_60_student.pth"
for NECK in after before; do
  "$PYTHON" -u "$ROOT/test.py" --config_file "$ROOT/configs/hp3_m25_a800.yml" \
    MODEL.PRETRAIN_CHOICE no M2.ENABLED False \
    TEST.WEIGHT "$OUT/transformer_60_student.pth" TEST.NECK_FEAT "$NECK" \
    DATASETS.PROTOCOL ALL OUTPUT_DIR "$OUT/eval_ALL_${NECK}"
done
