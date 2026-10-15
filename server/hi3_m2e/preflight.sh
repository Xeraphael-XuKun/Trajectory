#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory
# 只执行一次完整192图前后向及工件/部署核验，不启动60epoch。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/tools/hi3_m2e/check_combination.py --device cuda --batch-per-modality 64 --output /mnt/cache/wanghanzhi/XK/Trajectory_runs/hi3_m2e_1016/HI3_M2E_seed1234/preflight --config_file /mnt/cache/wanghanzhi/XK/hi3-m2e/Trajectory/configs/hi3_m2e/hi3_original_m2e.yml
