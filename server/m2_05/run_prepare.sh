#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/Trajectory-m2-e
PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
ROOT=/mnt/cache/wanghanzhi/Datasets/WHU-MARS
CLIP=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
ART=/mnt/cache/wanghanzhi/XK/outputs/m2-e/artifacts
mkdir -p "$ART"
$PY tools/m2/build_clip_cache.py --data-root "$ROOT" --clip "$CLIP" --out "$ART/train_clip_cache.pt"
$PY tools/m2/train_prompt_bank.py --cache "$ART/train_clip_cache.pt" --clip "$CLIP" --out "$ART/shared_id_prompts_final.pt" --steps 10000
$PY tools/m2/train_conditional_teacher.py --cache "$ART/train_clip_cache.pt" --text-bank "$ART/shared_id_prompts_final.pt" --clip "$CLIP" --out "$ART/conditional_teacher_epoch60.pth" --epochs 60
