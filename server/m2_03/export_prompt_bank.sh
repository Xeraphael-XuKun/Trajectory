#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-c/Trajectory
# 仅使用已完成的 deep_slot_prompt_bank.pt.stage_a.pt 重新导出；不新增训练更新。
# 旧版失败日志对应的脚本没有保存这个文件，不能用于恢复旧的10000步。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/server/m2_03/prompt_stage.py \
  --config-file /mnt/cache/wanghanzhi/XK/m2-c/Trajectory/configs/m2_03_deep_text_c0_control.yml \
  --export-only
