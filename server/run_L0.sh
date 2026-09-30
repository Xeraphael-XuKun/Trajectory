#!/usr/bin/env bash
# L0: llmpar with cudnn.benchmark=False.
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/Trajectory

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
export CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
export DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
export CONFIG=/mnt/cache/wanghanzhi/XK/Trajectory/configs/whu_trajectory_only.yml
export CUDNN_BENCHMARK=False
export OUTPUT_DIR=/mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_only_L0_llmpar_benchmark_false_seed1234

bash /mnt/cache/wanghanzhi/XK/Trajectory/run_trajectory.sh train

export CKPT=/mnt/cache/wanghanzhi/XK/Trajectory/logs/trajectory_only_L0_llmpar_benchmark_false_seed1234/transformer_60.pth
bash /mnt/cache/wanghanzhi/XK/Trajectory/run_trajectory.sh test
