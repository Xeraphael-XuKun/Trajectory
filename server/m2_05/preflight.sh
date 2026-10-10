#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-e/Trajectory
# 仅检查环境与配置，不执行训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -c 'import sys,torch,torchvision,timm,yacs,ftfy; from tools.m2.artifacts import load_config; c=load_config("/mnt/cache/wanghanzhi/XK/m2-e/Trajectory/configs/m2_05_conditional_teacher_distill.yml"); import model,processor; assert torch.cuda.is_available(); print(sys.version,torch.__version__,torchvision.__version__,timm.__version__); print(torch.cuda.get_device_name(0)); print("预检完成，未启动训练。")'
