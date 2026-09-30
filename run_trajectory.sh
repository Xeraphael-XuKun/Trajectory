#!/usr/bin/env bash
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
      TEST.WEIGHT "$CKPT"
    ;;
  *)
    echo "usage: bash run_trajectory.sh {train|test}" >&2
    exit 2
    ;;
esac
