# Trajectory + VTC 复现与消融

本项目使用 WHU-MARS、原始 OpenAI CLIP ViT-B/16 权重和单卡训练。
历史运行环境为 `/mnt/cache/wanghanzhi/envs/llmpar/bin/python3`
（Python 3.8）。本次复现保留该环境，不修改其中的软件包。

## 目标结果

`configs/whu_trajectory_vtc.yml` 在 seed 1234、固定 epoch 60 下的历史目标为：

- mAP：14.01%
- Rank-1：35.53%
- Rank-5：52.47%
- Rank-10：60.37%

这些数值是复现目标，不是代码运行前的保证。训练结束后必须通过磁盘中的
`transformer_60.pth` 独立重载评测。

## 服务器路径

```bash
cd /mnt/cache/wanghanzhi/XK/Trajectory

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
export CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
export DATA_ROOT=/mnt/cache/wanghanzhi/Datasets/WHU-MARS
```

## 实验一：Trajectory + VTC 复现

训练：

```bash
cd /mnt/cache/wanghanzhi/XK/Trajectory

CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 \
CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets/WHU-MARS \
CONFIG=configs/whu_trajectory_vtc.yml \
OUTPUT_DIR=./logs/repro_trajectory_vtc_seed1234 \
bash run_trajectory.sh train
```

固定 epoch-60 checkpoint 独立重载评测：

```bash
cd /mnt/cache/wanghanzhi/XK/Trajectory

CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 \
CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets/WHU-MARS \
CONFIG=configs/whu_trajectory_vtc.yml \
OUTPUT_DIR=./logs/repro_trajectory_vtc_seed1234 \
CKPT=./logs/repro_trajectory_vtc_seed1234/transformer_60.pth \
bash run_trajectory.sh test
```

## 实验二：Baseline + Trajectory（不加 VTC）

`configs/whu_trajectory_only.yml` 与复现配置保持同一数据、增强、采样、
优化器、学习率、epoch、seed、Trajectory 和评测协议，只关闭
`MODEL.TEXT_ALIGN` 并把 `SOLVER.TEXT_LOSS_WEIGHT` 设为 `0.0`。

训练：

```bash
cd /mnt/cache/wanghanzhi/XK/Trajectory

CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 \
CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets/WHU-MARS \
CONFIG=configs/whu_trajectory_only.yml \
OUTPUT_DIR=./logs/repro_trajectory_only_seed1234 \
bash run_trajectory.sh train
```

固定 epoch-60 checkpoint 独立重载评测：

```bash
cd /mnt/cache/wanghanzhi/XK/Trajectory

CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 \
CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets/WHU-MARS \
CONFIG=configs/whu_trajectory_only.yml \
OUTPUT_DIR=./logs/repro_trajectory_only_seed1234 \
CKPT=./logs/repro_trajectory_only_seed1234/transformer_60.pth \
bash run_trajectory.sh test
```

两组实验必须使用不同输出目录。不要使用 `torchrun` 或多卡 DDP；每条命令
都是一个独立的 `WORLD_SIZE=1` 前台任务。
