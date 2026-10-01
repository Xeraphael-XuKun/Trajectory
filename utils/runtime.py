import json
import os
import platform
import random
import sys
from pathlib import Path
import numpy as np
import timm
import torch
import torchvision


def configure_cudnn(benchmark=True, deterministic=True):
    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runtime_info():
    return {
        'executable': sys.executable, 'python': platform.python_version(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'timm': timm.__version__, 'numpy': np.__version__,
        'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic}


def runtime_summary():
    return 'Runtime: ' + json.dumps(runtime_info(), ensure_ascii=False)


def require_single_gpu():
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError('本轮只允许单卡：CUDA_VISIBLE_DEVICES=0，WORLD_SIZE=1')


def write_json(path, data):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare_run(cfg, training):
    require_single_gpu()
    if cfg.EXPERIMENT.PYTHON and os.name != 'nt' and os.path.abspath(sys.executable) != cfg.EXPERIMENT.PYTHON:
        raise RuntimeError('解释器与实验配置不一致：{}'.format(sys.executable))
    configure_cudnn(cfg.SOLVER.CUDNN_BENCHMARK, cfg.SOLVER.CUDNN_DETERMINISTIC)
    set_seed(cfg.SOLVER.SEED)
    out = Path(cfg.OUTPUT_DIR)
    if not cfg.OUTPUT_DIR or (out.exists() and any(out.iterdir())):
        raise RuntimeError('请使用空的独立 OUTPUT_DIR，避免混用旧 checkpoint 或日志')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'resolved_config.yml').open('w', encoding='utf-8') as stream:
        stream.write(cfg.dump())
    write_json(out / 'runtime.json', runtime_info())
    if cfg.TEST.NECK_FEAT not in ('before', 'after'):
        raise ValueError('NECK_FEAT must be before or after')
    if training and cfg.TEST.NECK_FEAT != 'before':
        raise ValueError('训练期检索主读出必须为 pre-BN / before')
