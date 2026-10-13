#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# 使用原epoch60教师做9图、FP32、一次临时学生更新及重载检查；临时产物自动删除。
# 不重训教师，不代替完整192图AMP检查或正式学生60epoch。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u "$REPO_DIR/tools/m2/check_reused_teacher.py" \
  --config-file "$REPO_DIR/configs/baseline_m2_05_reuse_teacher.yml" --device cuda
