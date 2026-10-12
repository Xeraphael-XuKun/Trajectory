"""Two complete student visual graphs, matched masks, one external RNG advance."""
import torch


def paired_visual_forward(base, images):
    devices = [images.device.index if images.device.index is not None else torch.cuda.current_device()] if images.is_cuda else []
    cpu_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(devices[0]) if devices else None
    on = base(images)
    # fork_rng saves post-on state and restores it even if OFF raises.
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(cpu_before)
        if devices: torch.cuda.set_rng_state(cuda_before, devices[0])
        off = base(images, trajectory_gate=images.new_zeros(len(images)))
    return on, off
