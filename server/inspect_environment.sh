#!/usr/bin/env bash
# 只检查已有解释器、包和 GPU/进程；不安装依赖、不启动训练。
set -euo pipefail
cd /mnt/cache/wanghanzhi/XK/Trajectory
nvidia-smi
ps -eo pid,user,args
for ENVIRONMENT in whu_mars llmpar; do
  PYTHON_BIN="/mnt/cache/wanghanzhi/envs/$ENVIRONMENT/bin/python3"
  OUTPUT="/mnt/cache/wanghanzhi/XK/Trajectory/logs/environment_1001/$ENVIRONMENT"
  mkdir -p "$OUTPUT"
  "$PYTHON_BIN" -m pip freeze --all > "$OUTPUT/pip_freeze.txt" 2>&1
  "$PYTHON_BIN" -m pip list --format=json > "$OUTPUT/pip_list.json" 2>&1
  # pip check 的已知不配套情况保留原始输出，不静默修复环境。
  set +e
  "$PYTHON_BIN" -m pip check > "$OUTPUT/pip_check.txt" 2>&1
  CHECK_STATUS=$?
  set -e
  echo "$CHECK_STATUS" > "$OUTPUT/pip_check_exit_code.txt"
  CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 "$PYTHON_BIN" -c \
    'from utils.runtime import configure_cudnn,runtime_info,write_json; import sys; configure_cudnn(True,True); write_json(sys.argv[1],runtime_info())' \
    "$OUTPUT/runtime.json" 2>&1 | tee -a "$OUTPUT/import_log.txt"
done
