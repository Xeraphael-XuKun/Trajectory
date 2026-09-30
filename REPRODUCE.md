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
export DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
```

## 实验一：Trajectory + VTC 复现

训练：

```bash
cd /mnt/cache/wanghanzhi/XK/Trajectory

CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
PYTHON_BIN=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 \
CLIP_WEIGHT=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets \
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
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets \
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
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets \
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
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets \
CONFIG=configs/whu_trajectory_only.yml \
OUTPUT_DIR=./logs/repro_trajectory_only_seed1234 \
CKPT=./logs/repro_trajectory_only_seed1234/transformer_60.pth \
bash run_trajectory.sh test
```

两组实验必须使用不同输出目录。不要使用 `torchrun` 或多卡 DDP；每条命令
都是一个独立的 `WORLD_SIZE=1` 前台任务。

## Trajectory-only 环境 × cuDNN 补充实验

现有 Trajectory-only 完整结果定义为 L1：`llmpar + benchmark=True`，不重复
运行。本轮只补齐下列三格；除解释器与 `CUDNN_BENCHMARK` 外，三组均使用
`configs/whu_trajectory_only.yml`、seed 1234、单卡和相同数据/权重。默认配置仍
保持历史行为 `benchmark=True`，每个启动器会显式覆盖并把实际运行环境写入日志。

| 编号 | Python 环境 | `cudnn.benchmark` | 输出目录 |
| --- | --- | --- | --- |
| L0 | `llmpar` | `False` | `logs/trajectory_only_L0_llmpar_benchmark_false_seed1234` |
| W0 | `whu_mars` | `False` | `logs/trajectory_only_W0_whu_mars_benchmark_false_seed1234` |
| W1 | `whu_mars` | `True` | `logs/trajectory_only_W1_whu_mars_benchmark_true_seed1234` |

三个脚本都在前台依次执行完整训练和 epoch-60 checkpoint 磁盘重载评测。
根据单卡资源实际空闲情况，每次只启动一条：

```bash
bash /mnt/cache/wanghanzhi/XK/Trajectory/server/run_L0_llmpar_benchmark_false.sh
```

```bash
bash /mnt/cache/wanghanzhi/XK/Trajectory/server/run_W0_whu_mars_benchmark_false.sh
```

```bash
bash /mnt/cache/wanghanzhi/XK/Trajectory/server/run_W1_whu_mars_benchmark_true.sh
```

日志中的 `Runtime:` 行会记录 Python、torch、torchvision、timm、torch CUDA、
cuDNN、GPU 型号以及 `cudnn_deterministic/cudnn_benchmark` 的实际值。比较时以
磁盘重载评测为主，并同时核对三组日志中的完整解析配置与这条运行时信息。
