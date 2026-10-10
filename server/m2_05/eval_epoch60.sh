#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 固定同一epoch60学生，分别输出post/pre-BN；不需要教师、文本或原CLIP工件。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/test.py --config_file /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml M2.ENABLED False MODEL.PRETRAIN_CHOICE no TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/logs/student/transformer_60_student.pth TEST.NECK_FEAT after OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-e/logs/eval_after
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/test.py --config_file /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml M2.ENABLED False MODEL.PRETRAIN_CHOICE no TEST.WEIGHT /mnt/cache/wanghanzhi/XK/m2-e/logs/student/transformer_60_student.pth TEST.NECK_FEAT before OUTPUT_DIR /mnt/cache/wanghanzhi/XK/m2-e/logs/eval_before
