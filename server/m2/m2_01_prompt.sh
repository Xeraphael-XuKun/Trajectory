#!/bin/sh
set -eu
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/Trajectory/tools/m2/train_condition_prompt.py --cache /mnt/cache/wanghanzhi/XK/m2-a/artifacts/train_clip_cache.pt --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt --output /mnt/cache/wanghanzhi/XK/m2-a/artifacts/condition_text_bank.pt --steps 10000
