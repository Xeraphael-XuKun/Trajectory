#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# 固定baseline+M2-5 epoch60学生；关闭训练期M2，不读取教师/文本/CLIP。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u "$REPO_DIR/test.py" \
  --config_file "$REPO_DIR/configs/baseline_m2_05_reuse_teacher.yml" \
  MODEL.TOKEN_TRAJECTORY False M2.ENABLED False MODEL.PRETRAIN_CHOICE no \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/baseline-m2-5/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/baseline-m2-5/logs/eval_after
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u "$REPO_DIR/test.py" \
  --config_file "$REPO_DIR/configs/baseline_m2_05_reuse_teacher.yml" \
  MODEL.TOKEN_TRAJECTORY False M2.ENABLED False MODEL.PRETRAIN_CHOICE no \
  TEST.WEIGHT /mnt/cache/wanghanzhi/XK/baseline-m2-5/logs/student/transformer_60_student.pth \
  TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/baseline-m2-5/logs/eval_before
