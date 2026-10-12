#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1
cd /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory
# 保存终端输出，并阻止覆盖已有正式D3结果。需要重跑时保留旧目录后修改输出路径。
if [[ -e /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server ]]; then
  echo 'D3_server已存在，请保留已有证据，修改本脚本的D3输出目录后再启动。' >&2
  exit 1
fi
mkdir -p /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server
exec > >(tee /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server/run.log) 2>&1
# D3为小型eval功能探针，不训练、不执行全测试集旁路评测。
# 两组使用同一510张清单；每个进程只导入对应工作树的原生模型代码。
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/scripts/r2/controller_probe.py select \
  --dataset /mnt/cache/wanghanzhi/Datasets/WHU-MARS \
  --output /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server --images 510 --seed 1234
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/scripts/r2/controller_probe.py probe \
  --role original --run-location server --batch-size 6 \
  --source-repo /mnt/cache/wanghanzhi/XK/m2-c/Trajectory \
  --checkpoint /mnt/cache/wanghanzhi/XK/m2-c/logs/student/transformer_60_student.pth \
  --dataset /mnt/cache/wanghanzhi/Datasets/WHU-MARS \
  --manifest /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server/fixed_train_images.csv \
  --output /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server/original
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/scripts/r2/controller_probe.py probe \
  --role e3 --run-location server --batch-size 6 \
  --source-repo /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory \
  --checkpoint /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3/transformer_60_student.pth \
  --full-checkpoint /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/E3/transformer_60.pth \
  --dataset /mnt/cache/wanghanzhi/Datasets/WHU-MARS \
  --manifest /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server/fixed_train_images.csv \
  --output /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server/e3
/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -u /mnt/cache/wanghanzhi/XK/m2-r2-e3/Trajectory/scripts/r2/controller_probe.py summarize \
  --output /mnt/cache/wanghanzhi/XK/m2-r2-e3/results/D3_server
