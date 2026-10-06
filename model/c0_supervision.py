"""Training-only paired C0 probes; no new parameters or inference branches."""
import torch
import torch.nn.functional as F


VARIANTS = ('c1', 'c1_absolute', 'c1_pooled', 'c2', 'c1_all3')
LOCAL_LAYERS = (7, 9, 11)  # zero-based; human blocks 8, 10, 12


def validate_c0_aux(cfg):
    if not cfg.C0_AUX.ENABLED:
        return
    if cfg.C0_AUX.VARIANT not in VARIANTS:
        raise ValueError('Unknown C0_AUX.VARIANT: ' + cfg.C0_AUX.VARIANT)
    if (not cfg.MODEL.TOKEN_TRAJECTORY or cfg.MODEL.TOKEN_TRAJECTORY_VARIANT != 'dense'
            or cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX != 0.0):
        raise ValueError('C0_AUX requires the original dense velocity-only C0')
    if (cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_clip'
            or cfg.MODEL.PE_TYPE != 'learnable' or cfg.MODEL.PE_LAYERWISE != 'none'
            or cfg.MODEL.PE_FREEZE_BASE or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW
            or cfg.MODEL.VPR or cfg.MODEL.MOD_DELTA or cfg.MODEL.TEXT_ALIGN
            or cfg.MODEL.CE_SPLIT_VIEW or cfg.MODEL.CE_SPLIT_MODALITY
            or cfg.MODEL.DIST_TRAIN or cfg.SOLVER.LOSS_TYPE != 'base'
            or cfg.SOLVER.TNCE_WEIGHT != 0 or cfg.SOLVER.TWIN_LOSS_WEIGHT != 0
            or cfg.SOLVER.MOD_DELTA_ONLY):
        raise ValueError('C0_AUX must use the single-GPU C0 + original CE/Triplet recipe')
    if (list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal']
            or list(cfg.DATASETS.AERIAL_CAMS) != [5, 6]
            or cfg.TEST.METRIC != 'sysu' or cfg.DATALOADER.SAMPLER != 'PKM'
            or cfg.DATALOADER.SYNC_FRAMES or cfg.DATALOADER.TIE_AUGMENTATION):
        raise ValueError('C0_AUX expects the original WHU-MARS modalities/cameras/PKM protocol')
    if (cfg.C0_AUX.WEIGHT <= 0 or cfg.C0_AUX.MAX_GAIN <= 0
            or cfg.C0_AUX.MARGIN < 0 or cfg.C0_AUX.KEEP_TOL < 0
            or cfg.C0_AUX.KEEP_WEIGHT < 0 or cfg.C0_AUX.WARMUP_EPOCHS < 0):
        raise ValueError('Invalid C0_AUX loss settings')


def probe_layers(variant, step):
    if variant == 'c2':
        return (9,)
    if variant == 'c1_all3':
        return LOCAL_LAYERS
    return (LOCAL_LAYERS[step % len(LOCAL_LAYERS)],)


def frozen_call(module, x):
    # Local import: disabled historical runs do not depend on this API.
    from torch.nn.utils.stateless import functional_call
    state = {name: value.detach() for name, value in module.named_parameters()}
    state.update({name: value.detach() for name, value in module.named_buffers()})
    return functional_call(module, state, (x,))


def paired_probe(base, bottleneck, state, layer, tail=False):
    """Only live C0 gains receive gradients; both branches replay main masks."""
    x, velocity, cpu_rng, cuda_rng = state
    device_ids = [x.device.index] if x.is_cuda else []
    layers = range(layer, len(base.blocks)) if tail else (layer,)

    def branch(enabled):
        z, v = x, velocity
        ratios = []
        with torch.random.fork_rng(devices=device_ids):
            torch.random.set_rng_state(cpu_rng)
            if x.is_cuda:
                torch.cuda.set_rng_state(cuda_rng, x.device)
            for index in layers:
                if enabled:
                    # Original C0 formula/class; keep recurrence gradients in C2.
                    correction = base.token_trajectory(index, v)
                    ratios.append((correction.detach().float().flatten(1).norm(dim=1)
                                   / z.detach().float().flatten(1).norm(dim=1).clamp_min(1e-12)).mean())
                    z = z + correction
                block_input = z
                z = frozen_call(base.blocks[index], z)
                v = z - block_input
            if tail:
                z = frozen_call(base.norm, z)[:, 0].float()
                # Read-only post-BN; never change buffers or the live train flag.
                with torch.cuda.amp.autocast(enabled=False):
                    z = F.batch_norm(z, bottleneck.running_mean.detach(),
                                     bottleneck.running_var.detach(),
                                     bottleneck.weight.detach(), bottleneck.bias.detach(),
                                     training=False, eps=bottleneck.eps)
            else:
                z = F.layer_norm(z[:, 0].float(), (z.shape[-1],))
            return F.normalize(z.float(), dim=-1), ratios

    with torch.no_grad():
        reference, _ = branch(False)
    corrected, ratios = branch(True)
    return reference, corrected, torch.stack(ratios).mean()


def build_probes(base, bottleneck, states, variant):
    return {layer: paired_probe(base, bottleneck, state, layer, tail=variant == 'c2')
            for layer, state in states.items()}
