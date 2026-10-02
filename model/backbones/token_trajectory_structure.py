"""Token-role and residual-source experiments; one historical dense gain each."""

import torch
import torch.nn.functional as F

from .token_trajectory import DenseCrossLayerTokenTrajectory
from .token_trajectory_variants import STRUCTURE_VARIANTS


class StructuredTokenTrajectory(DenseCrossLayerTokenTrajectory):
    def __init__(self, depth, num_tokens, embed_dim, variant,
                 acceleration_mix=1.0, hidden_dim=16, rank=16, ema_decay=0.25):
        super().__init__(depth, num_tokens, embed_dim, acceleration_mix)
        self.variant = variant
        self.register_buffer('method_signature', torch.tensor(
            [10 + STRUCTURE_VARIANTS.index(variant), depth, num_tokens,
             embed_dim, acceleration_mix, 0], dtype=torch.float64))

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        if self.variant in ('attention_velocity', 'mlp_velocity'):
            # The backbone supplies the selected, already-realized residual.
            return super().forward(layer, previous_velocity,
                                   previous_previous_velocity, gate)
        v, old = previous_velocity, previous_previous_velocity
        direction = v
        if old is not None:
            a = v - old
            if self.variant == 'cls_acceleration':
                direction = torch.cat((v[:, :1] + a[:, :1], v[:, 1:]), dim=1)
            else:  # patch_acceleration: CLS stays velocity-only.
                direction = torch.cat((v[:, :1], v[:, 1:] + a[:, 1:]), dim=1)
        direction = F.layer_norm(direction.float(), (self.embed_dim,)).to(v.dtype)
        correction = self.gain[layer - 1].to(v.dtype) * direction
        if gate is not None:
            correction = correction * gate.to(correction).reshape(-1, 1, 1)
        return correction
