"""Dense cross-layer token trajectory actuator for Vision Transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseCrossLayerTokenTrajectory(nn.Module):
    """Extrapolate each token from observed block velocity and acceleration.

    Unlike PLD, this module owns no static feature offset.  At layer ``l`` it
    observes the displacement produced by preceding Transformer blocks and
    applies a dense, learnable token/channel gain to that sample-specific
    trajectory direction.  The gain is exactly zero at initialisation.
    """

    def __init__(self, depth, num_tokens, embed_dim, acceleration_mix=1.0):
        super().__init__()
        depth = int(depth)
        num_tokens = int(num_tokens)
        embed_dim = int(embed_dim)
        if depth < 2:
            raise ValueError("token trajectory needs at least two layers")
        if num_tokens < 2 or embed_dim < 1:
            raise ValueError("invalid token trajectory shape")
        self.depth = depth
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.acceleration_mix = float(acceleration_mix)
        # Layer 0 has no preceding velocity.  Every later layer owns a dense
        # token/channel gain; no unused layer-0 parameter is allocated.
        self.gain = nn.Parameter(
            torch.zeros(depth - 1, 1, num_tokens, embed_dim)
        )

    @property
    def trajectory_parameters(self):
        return self.gain.numel()

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        if not 1 <= int(layer) < self.depth:
            raise IndexError(
                "trajectory layer {} outside [1, {})".format(layer, self.depth)
            )
        if previous_velocity.ndim != 3:
            raise ValueError("previous_velocity must be [batch, tokens, channels]")
        if previous_velocity.shape[1:] != (self.num_tokens, self.embed_dim):
            raise ValueError(
                "dense token trajectory was built for [{}, {}], got {}"
                .format(self.num_tokens, self.embed_dim,
                        tuple(previous_velocity.shape[1:]))
            )

        direction = previous_velocity
        if previous_previous_velocity is not None:
            acceleration = previous_velocity - previous_previous_velocity
            direction = direction + self.acceleration_mix * acceleration

        # Normalising only the observed direction prevents a large early block
        # displacement from setting the correction scale.  The zero-initialised
        # dense gain remains the sole learned magnitude.
        direction = F.layer_norm(
            direction.float(), (self.embed_dim,)
        ).to(dtype=previous_velocity.dtype)
        correction = self.gain[int(layer) - 1].to(
            dtype=previous_velocity.dtype
        ) * direction

        if gate is not None:
            gate = gate.to(
                device=correction.device, dtype=correction.dtype
            ).reshape(-1, 1, 1)
            correction = correction * gate
        return correction
