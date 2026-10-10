"""Independent Trajectory experiments; the historical dense class is unchanged."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_trajectory import DenseCrossLayerTokenTrajectory
from .token_trajectory_round5 import ROUND5_VARIANTS


VARIANT_CODES = {'adaptive': 1, 'split': 2, 'ema': 3,
                 'token_gate': 4, 'channel_mix': 5}
REFINEMENT_VARIANTS = ('dense_half', 'adaptive_half',
                       'token_gate_no_decay', 'velocity_gate')
STRUCTURE_VARIANTS = ('cls_acceleration', 'patch_acceleration',
                      'attention_velocity', 'mlp_velocity')
COMBINATION_VARIANTS = ('adaptive_token_gate', 'split_token_gate')
SPARSE_VARIANTS = ('velocity_cross_depth', 'velocity_late_depth')


def validate_trajectory_config(cfg):
    """Reject silently ignored knobs or mixed actuators in this experiment set."""
    variant = cfg.MODEL.TOKEN_TRAJECTORY_VARIANT
    if variant not in (('dense',) + tuple(VARIANT_CODES) + REFINEMENT_VARIANTS +
                       STRUCTURE_VARIANTS + COMBINATION_VARIANTS + SPARSE_VARIANTS +
                       ROUND5_VARIANTS):
        raise ValueError('Unknown Trajectory variant: {}'.format(variant))
    if variant != 'dense':
        if not cfg.MODEL.TOKEN_TRAJECTORY:
            raise ValueError('A Trajectory variant requires TOKEN_TRAJECTORY=True')
        expected_mix = {'dense_half': 0.5, 'velocity_gate': 0.0,
                        'attention_velocity': 0.0, 'mlp_velocity': 0.0,
                        'velocity_cross_depth': 0.0, 'velocity_late_depth': 0.0,
                        'velocity_uniform_transport': 0.0}.get(variant, 1.0)
        if cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX != expected_mix:
            raise ValueError('{} requires ACCEL_MIX={}'.format(variant, expected_mix))
        if cfg.MODEL.VPR or cfg.MODEL.MOD_DELTA or cfg.MODEL.TEXT_ALIGN:
            raise ValueError('Trajectory variants are independent: disable VPR, MOD_DELTA and TEXT_ALIGN')
    for owner, key, default in (
            ('ema', 'TOKEN_TRAJECTORY_EMA_DECAY', 0.25),
            ('token_gate', 'TOKEN_TRAJECTORY_HIDDEN_DIM', 16),
            ('channel_mix', 'TOKEN_TRAJECTORY_RANK', 16)):
        owners = (('token_gate', 'token_gate_no_decay', 'velocity_gate') + COMBINATION_VARIANTS
                  if owner == 'token_gate' else (owner,))
        if variant not in owners and getattr(cfg.MODEL, key) != default:
            raise ValueError('{} only affects variant {}'.format(key, owner))
    if variant in ROUND5_VARIANTS:
        requirements = (
            (cfg.MODEL, 'PE_TYPE', 'learnable'),
            (cfg.MODEL, 'PE_LAYERWISE', 'none'),
            (cfg.MODEL, 'PE_FREEZE_BASE', False),
            (cfg.MODEL, 'LAYER_SCALE', False),
            (cfg.MODEL, 'SIE_CAMERA', False),
            (cfg.MODEL, 'SIE_VIEW', False),
            (cfg.MODEL, 'CE_SPLIT_VIEW', False),
            (cfg.MODEL, 'CE_SPLIT_MODALITY', False),
            (cfg.MODEL, 'DIST_TRAIN', False),
            (cfg.MODEL, 'DROP_OUT', 0.0),
            (cfg.MODEL, 'ATT_DROP_RATE', 0.0),
            (cfg.C0_AUX, 'ENABLED', False),
            (cfg.TERMINAL_REPAIR, 'ENABLED', False),
            (cfg.SOLVER, 'LOSS_TYPE', 'base'),
            (cfg.SOLVER, 'TEXT_LOSS_WEIGHT', 0.0),
            (cfg.SOLVER, 'TWIN_LOSS_WEIGHT', 0.0),
            (cfg.SOLVER, 'TNCE_WEIGHT', 0.0),
            (cfg.SOLVER, 'MOD_DELTA_ONLY', False),
            (cfg.DATALOADER, 'SYNC_FRAMES', False),
            (cfg.DATALOADER, 'TIE_AUGMENTATION', False),
        )
        for node, key, expected in requirements:
            if getattr(node, key) != expected:
                raise ValueError('E3 requires {}={}'.format(key, expected))


class TrajectoryVariant(DenseCrossLayerTokenTrajectory):
    def __init__(self, depth, num_tokens, embed_dim, variant,
                 acceleration_mix=1.0, hidden_dim=16, rank=16, ema_decay=0.25):
        super().__init__(depth, num_tokens, embed_dim, acceleration_mix)
        self.variant = variant
        if variant == 'adaptive':
            # Ten acceleration-bearing positions, CLS/patch, bias/cos/log-RMS.
            self.beta_coeff = nn.Parameter(torch.zeros(depth - 2, 2, 3))
        elif variant == 'split':
            self.accel_gain = nn.Parameter(
                torch.zeros(depth - 2, 1, num_tokens, embed_dim))
        elif variant == 'token_gate':
            self.conditioners = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * embed_dim, hidden_dim), nn.GELU(),
                              nn.Linear(hidden_dim, 1))
                for _ in range(depth - 1)])
            for block in self.conditioners:
                nn.init.zeros_(block[-1].weight)
                nn.init.zeros_(block[-1].bias)
        elif variant == 'channel_mix':
            self.mixers = nn.ModuleList([
                nn.Sequential(nn.Linear(embed_dim, rank, bias=False), nn.GELU(),
                              nn.Linear(rank, embed_dim, bias=False))
                for _ in range(depth - 1)])
            for block in self.mixers:
                nn.init.zeros_(block[-1].weight)
        elif variant != 'ema':
            raise ValueError('Unknown Trajectory variant: {}'.format(variant))
        setting = {'ema': ema_decay, 'token_gate': hidden_dim,
                   'channel_mix': rank}.get(variant, 0)
        # A buffer, not a parameter: catches T0/T3 checkpoints with otherwise
        # identical tensor shapes. Historical dense checkpoints stay unchanged.
        self.register_buffer('method_signature', torch.tensor(
            [VARIANT_CODES[variant], depth, num_tokens, embed_dim,
             acceleration_mix, setting], dtype=torch.float64))

    @property
    def trajectory_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def normalize(self, value):
        return F.layer_norm(value.float(), (self.embed_dim,)).to(value.dtype)

    def adaptive_beta(self, layer, velocity, older):
        # Detached statistics control the predictor, while the direction's
        # velocity and acceleration retain their ordinary gradient paths.
        v, old = velocity.detach().float(), older.detach().float()
        cosine = F.cosine_similarity(v, old, dim=-1, eps=1e-6)
        rms = v.square().mean(dim=-1).sqrt()
        old_rms = old.square().mean(dim=-1).sqrt()
        ratio = torch.log((rms + 1e-6) / (old_rms + 1e-6)).tanh()
        weights = self.beta_coeff[layer - 2]
        weights = torch.cat((weights[:1], weights[1:].expand(self.num_tokens - 1, -1)))
        logits = weights[:, 0] + weights[:, 1] * cosine + weights[:, 2] * ratio
        return (2.0 * logits.sigmoid()).unsqueeze(-1).to(velocity.dtype)

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        # Smoothing is performed by the backbone's local velocity cache.
        if self.variant == 'ema':
            return super().forward(layer, previous_velocity,
                                   previous_previous_velocity, gate)
        v, old = previous_velocity, previous_previous_velocity
        a = None if old is None else v - old
        gain = self.gain[layer - 1].to(v.dtype)
        if self.variant == 'split':
            correction = gain * self.normalize(v)
            if a is not None:
                correction = correction + self.acceleration_mix * (
                    self.accel_gain[layer - 2].to(v.dtype) * self.normalize(a))
        else:
            direction = v
            if a is not None:
                beta = (self.adaptive_beta(layer, v, old)
                        if self.variant == 'adaptive' else self.acceleration_mix)
                direction = v + beta * a
            z = self.normalize(direction)
            correction = gain * z
            if self.variant == 'token_gate':
                a_norm = torch.zeros_like(v) if a is None else self.normalize(a)
                features = torch.cat((self.normalize(v), a_norm), dim=-1)
                logits = self.conditioners[layer - 1](features)
                multiplier = (0.5 + logits.float().sigmoid()).to(v.dtype)
                correction = correction * multiplier
            elif self.variant == 'channel_mix':
                correction = correction + self.mixers[layer - 1](z).to(v.dtype)
        if gate is not None:
            correction = correction * gate.to(correction).reshape(-1, 1, 1)
        return correction
