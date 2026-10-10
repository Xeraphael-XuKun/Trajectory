#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-a/Trajectory
# 读取实际版本、GPU 与完整合并配置；不启动训练。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -c 'import sys,torch,torchvision,timm,yacs,ftfy; from config import cfg; cfg.merge_from_file("configs/m2_01_condition_text_bridge.yml"); import model,processor; print(sys.version); print("torch",torch.__version__,"torchvision",torchvision.__version__,"timm",timm.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print("环境与配置预检完成，未启动训练。")'
