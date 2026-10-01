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


def write_json(path, data):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare_run(cfg):
    """应用设置并记录本次运行，不设置额外的启动前置门槛。"""
    configure_cudnn(cfg.SOLVER.CUDNN_BENCHMARK, cfg.SOLVER.CUDNN_DETERMINISTIC)
    set_seed(cfg.SOLVER.SEED)
    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'resolved_config.yml').open('w', encoding='utf-8') as stream:
        stream.write(cfg.dump())
    write_json(out / 'runtime.json', runtime_info())
