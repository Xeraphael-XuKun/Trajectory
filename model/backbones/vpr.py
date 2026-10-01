"""View-aware low-rank positional residuals (VPR).

VPR deliberately parameterises a *spatial field*, not a free ``N x D`` token
table.  Every layer owns ``rank`` channel vectors and combines them with a
fixed low-frequency 2-D DCT basis.  A sample-conditioned gate changes the
basis coefficients from image to image, while the external view gate decides
whether the field is active (aerial) or the zero reference (ground).

The coefficient bank and the conditioner are zero/identity initialised.  Thus
the complete module is exactly zero at step 0 and the pretrained CLIP path is
unchanged before optimisation.
"""

import math

import torch
import torch.nn as nn


def _dct_axis(length, frequency, device=None, dtype=torch.float32):
    """Orthonormal DCT-II vector of ``length`` for one frequency."""
    x = torch.arange(length, device=device, dtype=dtype)
    scale = math.sqrt(1.0 / length) if frequency == 0 else math.sqrt(2.0 / length)
    return scale * torch.cos(math.pi * (x + 0.5) * frequency / length)


def low_frequency_dct_basis(grid_h, grid_w, rank, device=None,
                            dtype=torch.float32):
    """Return ``[grid_h * grid_w, rank]`` non-DC low-frequency modes.

    Frequencies are ordered by total spatial frequency and then by their
    largest component.  The DC mode is excluded, so a correction cannot act as
    an image-wide semantic bias.  We centre again numerically and normalise
    each column because fp16 interpolation otherwise leaves a small mean.
    """
    n_patch = int(grid_h) * int(grid_w)
    if rank < 1 or rank > n_patch - 1:
        raise ValueError('VPR rank must be in [1, {}], got {}'.format(
            n_patch - 1, rank))
    pairs = [(fy, fx) for fy in range(grid_h) for fx in range(grid_w)
             if fy != 0 or fx != 0]
    pairs.sort(key=lambda p: (p[0] + p[1], max(p), p[0], p[1]))
    cols = []
    for fy, fx in pairs[:rank]:
        by = _dct_axis(grid_h, fy, device=device, dtype=dtype)
        bx = _dct_axis(grid_w, fx, device=device, dtype=dtype)
        col = torch.outer(by, bx).reshape(-1)
        col = col - col.mean()
        col = col / col.norm().clamp_min(torch.finfo(col.dtype).eps)
        cols.append(col)
    return torch.stack(cols, dim=1)


class ViewAwarePositionalResidual(nn.Module):
    """Continuous, low-rank, sample-adaptive layer-wise position correction.

    Args:
        depth: number of Transformer blocks.
        embed_dim: token channel dimension.
        grid_h/grid_w: training patch grid.
        rank: number of non-DC DCT modes.
        conditional: if true, predict a per-image multiplier for every mode
            from the current CLS token.  The multiplier is one at step 0.
        zero_mean: remove the spatial mean after reconstruction as a strict
            guard against turning VPR into a global feature bias.
    """

    def __init__(self, depth, embed_dim, grid_h, grid_w, rank=16,
                 conditional=True, zero_mean=True):
        super().__init__()
        self.depth = int(depth)
        self.embed_dim = int(embed_dim)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.rank = int(rank)
        self.conditional = bool(conditional)
        self.zero_mean = bool(zero_mean)

        basis = low_frequency_dct_basis(self.grid_h, self.grid_w, self.rank)
        self.register_buffer('basis', basis, persistent=False)
        # [layer, spatial mode, token channel].  This is the only field bank.
        self.coeff = nn.Parameter(torch.zeros(
            self.depth, self.rank, self.embed_dim))

        if self.conditional:
            self.condition_norm = nn.LayerNorm(self.embed_dim)
            self.condition = nn.Linear(self.embed_dim, self.rank)
            # 2*sigmoid(0) == 1: sample conditioning starts as an identity.
            nn.init.zeros_(self.condition.weight)
            nn.init.zeros_(self.condition.bias)
        else:
            self.condition_norm = None
            self.condition = None

    def _basis(self, grid_h, grid_w, reference):
        if int(grid_h) == self.grid_h and int(grid_w) == self.grid_w:
            return self.basis.to(device=reference.device, dtype=reference.dtype)
        # Analytic coordinates make the residual continuous across resolutions;
        # no learned table is resized or extrapolated.
        return low_frequency_dct_basis(
            int(grid_h), int(grid_w), self.rank,
            device=reference.device, dtype=torch.float32).to(reference.dtype)

    def forward(self, layer, cls_token, grid_h, grid_w):
        """Reconstruct one layer's patch field as ``[B, N, D]``.

        CLS is used only to condition the patch field and never receives a
        residual row.  This makes the claim "positional" mechanically testable.
        """
        if layer < 0 or layer >= self.depth:
            raise IndexError('VPR layer {} outside [0, {})'.format(
                layer, self.depth))
        basis = self._basis(grid_h, grid_w, cls_token)
        if self.condition is None:
            gate = cls_token.new_ones(cls_token.shape[0], self.rank)
        else:
            gate = 2.0 * torch.sigmoid(
                self.condition(self.condition_norm(cls_token)))
        coeff = self.coeff[layer].to(cls_token.dtype)
        field = torch.einsum('nr,br,rd->bnd', basis, gate, coeff)
        if self.zero_mean:
            field = field - field.mean(dim=1, keepdim=True)
        return field

    @property
    def field_parameters(self):
        """Count only the layer/mode/channel correction bank."""
        return self.coeff.numel()
