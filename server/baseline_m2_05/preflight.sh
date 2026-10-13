#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
# 以脚本所在仓库定位代码，clone目录名可以改变；输出和原教师来源保持明确。
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# 只严格核对已有教师/文本/prompt/训练来源，不更新教师或学生。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u "$REPO_DIR/tools/m2/check_reused_teacher.py" \
  --config-file "$REPO_DIR/configs/baseline_m2_05_reuse_teacher.yml" --device cuda --artifacts-only
