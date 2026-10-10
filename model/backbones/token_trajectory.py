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


class DeepTextC0Controller(nn.Module):
    """Positive amplitude modulation; one owner of the shared C_l bank."""
    def __init__(self, depth, num_tokens, embed_dim=768, context_dim=512,
                 n_ctx=4, width=64, rho=0.25):
        super().__init__()
        self.depth, self.num_tokens, self.rho = int(depth), int(num_tokens), float(rho)
        self.context = nn.Parameter(torch.zeros(depth - 1, n_ctx, context_dim))
        self.q = nn.Linear(embed_dim, width, bias=False)
        self.k = nn.Linear(context_dim, width, bias=False)
        self.v = nn.Linear(context_dim, width, bias=False)
        self.o = nn.Linear(width, embed_dim, bias=False)
        nn.init.zeros_(self.o.weight)
        self.last_stats = []

    def key_values(self):
        # Once per image batch, attached to autograd during training.
        with torch.cuda.amp.autocast(enabled=False):
            c = F.layer_norm(self.context.float(), (self.context.shape[-1],))
            return self.k(c), self.v(c)

    def forward(self, layer, velocity, correction, key_values=None):
        with torch.cuda.amp.autocast(enabled=False):
            layer = int(layer) - 1
            key, value = self.key_values() if key_values is None else key_values
            u = F.layer_norm(velocity.float(), (velocity.shape[-1],))
            attention = torch.softmax(self.q(u) @ key[layer].t() / self.q.out_features ** 0.5, dim=-1)
            multiplier = 1 + self.rho * torch.tanh(self.o(attention @ value[layer]))
            result = correction * multiplier.to(correction.dtype)
            if self.training:
                with torch.no_grad():
                    ratio = result.float().norm() / correction.float().norm().clamp_min(1e-12)
                    self.last_stats.append(torch.stack([multiplier.mean(), multiplier.std(),
                                                        multiplier.min(), multiplier.max(), ratio]))
            return result
