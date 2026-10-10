"""E3: uniform transport of full-block velocity (round-five version 1)."""

import torch
import torch.nn.functional as F

from .token_trajectory import DenseCrossLayerTokenTrajectory


# Only E3 is registered here; E1 and E2 have independent worktrees.
ROUND5_CODES = {"velocity_uniform_transport": 703}
ROUND5_VARIANTS = tuple(ROUND5_CODES)
ROUND5_IMPLEMENTATION_VERSION = 1


class Round5VelocityTrajectory(DenseCrossLayerTokenTrajectory):
    def __init__(self, depth, num_tokens, embed_dim, variant,
                 acceleration_mix=0.0, hidden_dim=16, rank=16, ema_decay=0.25):
        if variant not in ROUND5_CODES:
            raise ValueError("This branch implements velocity_uniform_transport (E3) only")
        if float(acceleration_mix) != 0.0:
            raise ValueError("E3 requires ACCEL_MIX=0.0")
        if hidden_dim != 16 or rank != 16 or float(ema_decay) != 0.25:
            raise ValueError("E3 does not use HIDDEN_DIM/RANK/EMA_DECAY")
        super().__init__(depth, num_tokens, embed_dim, acceleration_mix=0.0)
        self.variant = variant
        self.transport_weight = 0.5
        self.norm_eps = 1e-5
        # code, version, L, T, D, accel_mix, transport_weight, transient,
        # detach_attention, fp32_direction, LN epsilon; buffer, not a parameter.
        self.register_buffer("method_signature", torch.tensor([
            703, ROUND5_IMPLEMENTATION_VERSION, self.depth, self.num_tokens,
            self.embed_dim, 0.0, 0.5, 0.0, 0.0, 1.0, self.norm_eps,
        ], dtype=torch.float64))

    def normalized_direction(self, previous_velocity):
        v = previous_velocity
        if v.ndim != 3 or tuple(v.shape[1:]) != (self.num_tokens, self.embed_dim):
            raise ValueError("Velocity must be [B, {}, {}]".format(
                self.num_tokens, self.embed_dim))
        if not v.is_floating_point():
            raise TypeError("Velocity must be floating point")
        # Both the local and mean paths remain differentiable. The mean is
        # per image, includes CLS, and needs no dense uniform matrix.
        with torch.autocast(device_type=v.device.type, enabled=False):
            v32 = v.float()
            mean_velocity = v32.mean(dim=1, keepdim=True)
            mixed = 0.5 * v32 + 0.5 * mean_velocity
            direction = F.layer_norm(mixed, (self.embed_dim,),
                                     weight=None, bias=None, eps=self.norm_eps)
        return direction.to(dtype=v.dtype)

    def forward(self, layer, previous_velocity, previous_previous_velocity=None,
                gate=None):
        index = int(layer)
        if index != layer or not 1 <= index < self.depth:
            raise IndexError("Invalid zero-based injection index: {}".format(layer))
        if previous_previous_velocity is not None:
            raise ValueError("E3 uses one previous full-block velocity")
        if self.gain.device != previous_velocity.device:
            raise ValueError("Trajectory and velocity must share device")
        direction = self.normalized_direction(previous_velocity)
        correction = self.gain[index - 1].to(previous_velocity.dtype) * direction
        if gate is not None:
            if gate.numel() != previous_velocity.shape[0]:
                raise ValueError("Gate must contain exactly one value per image")
            correction = correction * gate.to(correction).reshape(-1, 1, 1)
        return correction
