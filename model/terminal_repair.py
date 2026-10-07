"""Training-only helpers for full-C0 terminal comparisons. No model mutation."""
import torch
import torch.nn.functional as F


def validate_terminal_repair(cfg):
    if not cfg.TERMINAL_REPAIR.ENABLED:
        return
    r = cfg.TERMINAL_REPAIR
    if (r.WEIGHTING not in ('uniform', 'reference', 'self')
            or r.GRADIENT_SCOPE not in ('joint', 'gain_only')
            or r.WEIGHT <= 0 or r.TEMPERATURE <= 0 or r.WARMUP_EPOCHS < 0):
        raise ValueError('Invalid TERMINAL_REPAIR settings')
    if (cfg.C0_AUX.ENABLED or not cfg.MODEL.TOKEN_TRAJECTORY
            or cfg.MODEL.TOKEN_TRAJECTORY_VARIANT != 'dense'
            or cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX != 0.0
            or cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_clip'
            or cfg.MODEL.PE_TYPE != 'learnable' or cfg.MODEL.PE_LAYERWISE != 'none'
            or cfg.MODEL.PE_FREEZE_BASE or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW
            or cfg.MODEL.VPR or cfg.MODEL.MOD_DELTA or cfg.MODEL.TEXT_ALIGN
            or cfg.MODEL.CE_SPLIT_VIEW or cfg.MODEL.CE_SPLIT_MODALITY
            or cfg.MODEL.DIST_TRAIN or cfg.SOLVER.LOSS_TYPE != 'base'
            or cfg.SOLVER.TNCE_WEIGHT != 0 or cfg.SOLVER.TWIN_LOSS_WEIGHT != 0
            or cfg.SOLVER.MOD_DELTA_ONLY):
        raise ValueError('TERMINAL_REPAIR requires single-GPU C0 + original CE/Triplet only')
    if (list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal']
            or list(cfg.DATASETS.AERIAL_CAMS) != [5, 6] or cfg.TEST.METRIC != 'sysu'
            or cfg.DATALOADER.SAMPLER != 'PKM' or cfg.DATALOADER.SYNC_FRAMES
            or cfg.DATALOADER.TIE_AUGMENTATION):
        raise ValueError('TERMINAL_REPAIR expects the original WHU-MARS PKM protocol')


def rng_state(device):
    return torch.random.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None


def set_rng(state, device):
    torch.random.set_rng_state(state[0])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state[1], device)


def readonly_neck(features, bottleneck):
    with torch.cuda.amp.autocast(enabled=False):
        x = F.batch_norm(features.float(), bottleneck.running_mean.detach(),
                         bottleneck.running_var.detach(), bottleneck.weight.detach(),
                         bottleneck.bias.detach(), training=False, eps=bottleneck.eps)
        return F.normalize(x, dim=-1)


def replay(base, images, state, enabled=True, gain_only=False):
    """Replay the input RNG, then restore caller RNG; preserve full recurrence."""
    devices = [images.device.index] if images.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        set_rng(state, images.device)
        kwargs = {} if enabled else {'trajectory_gate': images.new_zeros(images.shape[0])}
        if gain_only:
            from torch.nn.utils.stateless import functional_call
            params = {n: p if n.startswith('token_trajectory.') else p.detach()
                      for n, p in base.named_parameters()}
            params.update({n: b.detach() for n, b in base.named_buffers()})
            return functional_call(base, params, (images,), kwargs)
        return base(images, **kwargs)
