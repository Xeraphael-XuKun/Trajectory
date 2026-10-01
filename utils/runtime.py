import platform

import timm
import torch
import torchvision


def configure_cudnn(benchmark):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = bool(benchmark)


def runtime_summary():
    device = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else 'cpu'
    )
    return (
        'Runtime: '
        f'python={platform.python_version()}, '
        f'torch={torch.__version__}, '
        f'torchvision={torchvision.__version__}, '
        f'timm={timm.__version__}, '
        f'torch_cuda={torch.version.cuda}, '
        f'cudnn={torch.backends.cudnn.version()}, '
        f'device={device}, '
        f'cudnn_deterministic={torch.backends.cudnn.deterministic}, '
        f'cudnn_benchmark={torch.backends.cudnn.benchmark}'
    )
