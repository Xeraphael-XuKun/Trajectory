#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 只读train，原始冻结CLIP，确定性三光谱特征。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/tools/m2/build_clip_cache.py --config-file /mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml --device cuda --batch-size 64
