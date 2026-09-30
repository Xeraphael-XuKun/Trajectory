#!/usr/bin/env bash
# W1: whu_mars with cudnn.benchmark=True.
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/Trajectory

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHON_BIN=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
export CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
export DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
export CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory/configs/whu_trajectory_only.yml
export CUDNN_BENCHMARK=True
export OUTPUT_DIR=/mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_only_W1_whu_mars_benchmark_true_seed1234

bash /mnt/cache/wanghanzhi/XK/Trajectory/run_trajectory.sh train

export CKPT=/mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_only_W1_whu_mars_benchmark_true_seed1234/transformer_60.pth
bash /mnt/cache/wanghanzhi/XK/Trajectory/run_trajectory.sh test
