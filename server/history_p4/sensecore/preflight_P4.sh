#!/usr/bin/env bash
# P4：单张 A800、192 图 AMP 预检；此脚本不启动正式 60 epoch 实验。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory_history_p4
RUN_DIR=/mnt/cache/wanghanzhi/XK/Trajectory_runs/history_round2_1016/P4_preflight_seed1234
if [[ -e "$RUN_DIR" ]]; then echo "预检目录已存在，请使用独立目录：$RUN_DIR" >&2; exit 1; fi
# Python 工具创建目录，保留原 60 epoch 日程；只重复一个真实 batch。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u \
  /mnt/cache/wanghanzhi/XK/Trajectory_history_p4/tools/smoke_history_p4.py \
  --pretrained /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --batch-size 64 --num-instance 4 --steps 16 --output "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
