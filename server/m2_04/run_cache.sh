#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-d/Trajectory
# 仅train/RGB原始冻结CLIP；同一命令输出特征/相机均衡中心/余弦关系。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/tools/build_m2_rgb_centers.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-d/Trajectory/configs/m2_04_clip_relation_distill.yml \
  --device cuda --batch-size 64 --workers 8
