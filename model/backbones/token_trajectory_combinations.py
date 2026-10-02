"""Round four: original T1/T2 directions with T4 gating, or sparse C0."""

import torch
from torch import nn
import torch.nn.functional as F

from .token_trajectory_variants import TrajectoryVariant


COMBINATION_CODES = {'adaptive_token_gate': 14, 'split_token_gate': 15}
SPARSE_BLOCKS = {'velocity_cross_depth': (5, 8, 12),
                 'velocity_late_depth': (10, 11, 12)}


class GatedTrajectoryCombination(TrajectoryVariant):
    def __init__(self, depth, num_tokens, embed_dim, variant,
                 acceleration_mix=1.0, hidden_dim=16, rank=16, ema_decay=0.25):
        direction_variant = ('adaptive' if variant == 'adaptive_token_gate' else 'split')
        super().__init__(depth, num_tokens, embed_dim, direction_variant,
                         acceleration_mix, hidden_dim, rank, ema_decay)
        self.variant = variant
        # Exactly the original T4 conditioner, unit multiplier at initialization.
        self.conditioners = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * embed_dim, hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, 1)) for _ in range(depth - 1)])
        for block in self.conditioners:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
        self.method_signature = torch.tensor(
            [COMBINATION_CODES[variant], depth, num_tokens, embed_dim,
             acceleration_mix, hidden_dim], dtype=torch.float64)

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        v, old = previous_velocity, previous_previous_velocity
        a = None if old is None else v - old
        gain = self.gain[layer - 1].to(v.dtype)
        if self.variant == 'adaptive_token_gate':
            direction = v if a is None else v + self.adaptive_beta(layer, v, old) * a
            correction = gain * self.normalize(direction)
        else:
            correction = gain * self.normalize(v)
            if a is not None:
                correction = correction + self.acceleration_mix * (
                    self.accel_gain[layer - 2].to(v.dtype) * self.normalize(a))
        a_norm = torch.zeros_like(v) if a is None else self.normalize(a)
        features = torch.cat((self.normalize(v), a_norm), dim=-1)
        logits = self.conditioners[layer - 1](features)
        multiplier = (0.5 + logits.float().sigmoid()).to(v.dtype)
        # One gate acts on the complete correction, including BOTH T2 branches.
        correction = correction * multiplier
        if gate is not None:
            correction = correction * gate.to(correction).reshape(-1, 1, 1)
        return correction


class SparseVelocityTrajectory(nn.Module):
    def __init__(self, depth, num_tokens, embed_dim, variant,
                 acceleration_mix=0.0, hidden_dim=16, rank=16, ema_decay=0.25):
        super().__init__()
        self.variant = variant
        self.acceleration_mix = float(acceleration_mix)
        self.embed_dim = embed_dim
        # User-facing block numbers are one-based; backbone calls are zero-based.
        self.active_blocks = SPARSE_BLOCKS[variant]
        if max(self.active_blocks) > depth or acceleration_mix != 0.0:
            raise ValueError('Sparse C0 requires the selected blocks and ACCEL_MIX=0')
        self.active_layers = tuple(block - 1 for block in self.active_blocks)
        self.layer_to_row = {layer: row for row, layer in enumerate(self.active_layers)}
        self.gain = nn.Parameter(torch.zeros(len(self.active_layers), 1, num_tokens, embed_dim))
        code = 16 if variant == 'velocity_cross_depth' else 17
        self.register_buffer('method_signature', torch.tensor(
            [code, depth, num_tokens, embed_dim, acceleration_mix, 0,
             *self.active_blocks], dtype=torch.float64))

    @property
    def trajectory_parameters(self):
        return self.gain.numel()

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        v = previous_velocity
        direction = F.layer_norm(v.float(), (self.embed_dim,)).to(v.dtype)
        correction = self.gain[self.layer_to_row[layer]].to(v.dtype) * direction
        if gate is not None:
            correction = correction * gate.to(correction).reshape(-1, 1, 1)
        return correction
