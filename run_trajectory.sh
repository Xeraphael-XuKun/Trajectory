#!/usr/bin/env bash
# Shared Linux entrypoint for Trajectory training and checkpoint evaluation.
set -euo pipefail

# Usage:
#   CLIP_WEIGHT=/path/ViT-B-16.pt DATA_ROOT=/path/to/data bash run_trajectory.sh train
#   CKPT=/path/transformer_60.pth DATA_ROOT=/path/to/data bash run_trajectory.sh test

ACTION=${1:-train}
PYTHON_BIN=${PYTHON_BIN:-python3}
CLIP_WEIGHT=${CLIP_WEIGHT:-ViT-B-16.pt}
DATA_ROOT=${DATA_ROOT:-/data}
OUTPUT_DIR=${OUTPUT_DIR:-./logs/whu_trajectory_vtc}
CONFIG=${CONFIG:-configs/whu_trajectory_vtc.yml}
CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-True}

# YACS parses command-line overrides as Python literals.  Passing shell-style
# lowercase true/false leaves a string and fails its bool type check, so convert
# accepted spellings to the exact literals YACS expects before building opts.
case "$CUDNN_BENCHMARK" in
  true|True|TRUE)
    CUDNN_BENCHMARK_YACS=True
    ;;
  false|False|FALSE)
    CUDNN_BENCHMARK_YACS=False
    ;;
  *)
    echo "CUDNN_BENCHMARK must be True or False, got: $CUDNN_BENCHMARK" >&2
    exit 2
    ;;
esac

if [ ! -f "$CLIP_WEIGHT" ]; then
  echo "missing official CLIP checkpoint: $CLIP_WEIGHT" >&2
  exit 1
fi

case "$ACTION" in
  train)
    "$PYTHON_BIN" -u train.py --config_file "$CONFIG" \
      MODEL.PRETRAIN_PATH "$CLIP_WEIGHT" \
      MODEL.TEXT_CLIP_PATH "$CLIP_WEIGHT" \
      DATASETS.ROOT_DIR "$DATA_ROOT" \
      SOLVER.CUDNN_BENCHMARK "$CUDNN_BENCHMARK_YACS" \
      OUTPUT_DIR "$OUTPUT_DIR"
    ;;
  test)
    CKPT=${CKPT:-$OUTPUT_DIR/transformer_60.pth}
    if [ ! -f "$CKPT" ]; then
      echo "missing checkpoint: $CKPT" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u test.py --config_file "$CONFIG" \
      MODEL.PRETRAIN_PATH "$CLIP_WEIGHT" \
      MODEL.TEXT_CLIP_PATH "$CLIP_WEIGHT" \
      DATASETS.ROOT_DIR "$DATA_ROOT" \
      SOLVER.CUDNN_BENCHMARK "$CUDNN_BENCHMARK_YACS" \
      TEST.WEIGHT "$CKPT" \
      OUTPUT_DIR "$OUTPUT_DIR"
    ;;
  *)
    echo "usage: bash run_trajectory.sh {train|test}" >&2
    exit 2
    ;;
esac
