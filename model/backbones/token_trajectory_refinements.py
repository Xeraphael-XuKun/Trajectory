"""Round-two variants, with distinct checkpoint identities and old defaults intact."""

import math

import torch
import torch.nn as nn

from .token_trajectory import DenseCrossLayerTokenTrajectory
from .token_trajectory_variants import TrajectoryVariant, REFINEMENT_VARIANTS


class TrajectoryRefinement(TrajectoryVariant):
    def __init__(self, depth, num_tokens, embed_dim, variant,
                 acceleration_mix=1.0, hidden_dim=16, rank=16, ema_decay=0.25):
        self.experiment_variant = variant
        if variant in ('dense_half', 'velocity_gate'):
            DenseCrossLayerTokenTrajectory.__init__(
                self, depth, num_tokens, embed_dim, acceleration_mix)
            self.variant = variant
            if variant == 'velocity_gate':
                self.conditioners = nn.ModuleList([
                    nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, 1))
                    for _ in range(depth - 1)])
                for block in self.conditioners:
                    nn.init.zeros_(block[-1].weight)
                    nn.init.zeros_(block[-1].bias)
        else:
            base_variant = {'adaptive_half': 'adaptive',
                            'token_gate_no_decay': 'token_gate'}[variant]
            super().__init__(depth, num_tokens, embed_dim, base_variant,
                             acceleration_mix, hidden_dim, rank, ema_decay)
            if variant == 'adaptive_half':
                # 2 * sigmoid(-log(3)) = 0.5; conditioning slopes remain zero.
                with torch.no_grad():
                    self.beta_coeff[:, :, 0].fill_(-math.log(3.0))

        self.gate_no_weight_decay = variant in ('token_gate_no_decay', 'velocity_gate')
        setting = (hidden_dim if self.gate_no_weight_decay else
                   0.5 if variant == 'adaptive_half' else 0.0)
        signature = torch.tensor([6 + REFINEMENT_VARIANTS.index(variant),
                                  depth, num_tokens, embed_dim,
                                  acceleration_mix, setting], dtype=torch.float64)
        if 'method_signature' in self._buffers:
            self.method_signature = signature
        else:
            self.register_buffer('method_signature', signature)

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        if self.variant == 'dense_half':
            return DenseCrossLayerTokenTrajectory.forward(
                self, layer, previous_velocity, previous_previous_velocity, gate)
        if self.variant == 'velocity_gate':
            # Both direction and conditioner are velocity-only. No acceleration
            # enters indirectly via the gate; at initialization multiplier=1.
            v = previous_velocity
            z = self.normalize(v)
            logits = self.conditioners[layer - 1](z)
            multiplier = (0.5 + logits.float().sigmoid()).to(v.dtype)
            correction = self.gain[layer - 1].to(v.dtype) * z * multiplier
            if gate is not None:
                correction = correction * gate.to(correction).reshape(-1, 1, 1)
            return correction
        return super().forward(layer, previous_velocity, previous_previous_velocity, gate)
