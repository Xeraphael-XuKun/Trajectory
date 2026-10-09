#!/usr/bin/env bash
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/m2-b/Trajectory
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -m py_compile model/m2_trajectory_inverter.py loss/m2_losses.py tools/m2/build_clip_cache.py tools/m2/build_rgb_center_cache.py model/make_model.py model/backbones/vit_pytorch.py processor/processor.py
echo 'M2-2 静态预检完成；未启动训练。'


