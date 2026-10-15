"""P2: detached history prediction and patch-grid spatial innovation adaptation."""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryP2Adapter(nn.Module):
    def __init__(self, dim, depth, patches, rank=64, alpha=0.25,
                 prediction_layers=None, grid_size=(16, 8)):
        super().__init__()
        self.alpha = float(alpha)
        self.grid_h, self.grid_w = grid_size
        if patches != self.grid_h * self.grid_w:
            raise ValueError('P2 patch count must match the spatial grid')
        self.token_scope, self.gain_mode = 'patch', 'linear'
        self.prediction_layers = frozenset(range(2, depth + 1))
        if prediction_layers is not None and set(prediction_layers) != self.prediction_layers:
            raise ValueError('P2 predicts at every block from 2 through depth')
        # Same first random draw and QR as P1 under the isolated INIT_SEED.
        coordinate, _ = torch.linalg.qr(torch.randn(dim, rank), mode='reduced')
        self.register_buffer('coordinate', coordinate)
        self.register_buffer('specification', torch.tensor(
            [102, dim, depth, patches, rank, alpha, self.grid_h, self.grid_w]))
        self.gain = nn.Parameter(torch.zeros(depth - 1, 1, patches, dim))
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * rank, rank), nn.GELU(), nn.Linear(rank, rank))
            for _ in range(depth - 1)])
        for predictor in self.predictors:
            nn.init.zeros_(predictor[-1].weight)
            nn.init.zeros_(predictor[-1].bias)
        self.fusions = nn.ModuleList([nn.Linear(3 * rank, rank) for _ in range(depth - 1)])
        self.depthwise3 = nn.ModuleList([
            nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False)
            for _ in range(depth - 1)])
        self.depthwise5 = nn.ModuleList([
            nn.Conv2d(rank, rank, 5, padding=2, groups=rank, bias=False)
            for _ in range(depth - 1)])
        for convolution in list(self.depthwise3) + list(self.depthwise5):
            nn.init.zeros_(convolution.weight)
        self.up_projections = nn.ModuleList([nn.Linear(rank, dim) for _ in range(depth - 1)])

    def correction(self, h, layer, history, previous_r, collect=False, diagnostics=False):
        with torch.autocast(device_type=h.device.type, enabled=False):
            patches = h[:, 1:].float()
            s = F.layer_norm(patches, (patches.shape[-1],), eps=1e-5) @ self.coordinate
            prediction = target = None
            if layer == 0:
                r = torch.zeros_like(h)
                next_r = torch.zeros_like(s)
            else:
                index = layer - 1
                last = history[-1]
                old_delta = last - history[-2] if len(history) > 1 else torch.zeros_like(last)
                prediction = self.predictors[index](torch.cat((last, old_delta), dim=-1))
                innovation = s - (last + prediction).detach()
                fused = innovation + F.gelu(self.fusions[index](torch.cat((s, innovation, previous_r), -1)))
                # PatchEmbed flattens height then width; restore exactly that ordering.
                grid = fused.transpose(1, 2).reshape(h.shape[0], -1, self.grid_h, self.grid_w)
                spatial_update = 0.5 * (self.depthwise3[index](grid) + self.depthwise5[index](grid))
                mixed = (grid + spatial_update).flatten(2).transpose(1, 2)
                direction = self.up_projections[index](F.gelu(mixed))
                normalized_direction = F.layer_norm(direction, (patches.shape[-1],), eps=1e-5)
                patch_r = self.alpha * self.gain[index] * normalized_direction
                r = torch.cat((torch.zeros_like(h[:, :1]), patch_r.to(h.dtype)), dim=1)
                next_r = r[:, 1:].detach().float() @ self.coordinate
                if collect:
                    target = (s - last).detach()
            next_history = (history + [s.detach()])[-2:]
            item = None
            if collect:
                item = {'prediction': prediction, 'target': target}
                if diagnostics:
                    with torch.no_grad():
                        actual_r = r[:, 1:].float()
                        stats = {
                            'state_variance': s.var(-1, unbiased=False).mean(),
                            'correction_rms': actual_r.square().mean().sqrt(),
                            'correction_ratio': actual_r.square().mean().sqrt() / patches.square().mean().sqrt().clamp_min(1e-8),
                        }
                        if layer > 0:
                            stats.update(
                                innovation_rms=innovation.square().mean().sqrt(),
                                spatial_update_rms=spatial_update.square().mean().sqrt(),
                                direction_rms=direction.square().mean().sqrt(),
                                normalized_direction_rms=normalized_direction.square().mean().sqrt(),
                                prediction_mse=(prediction - target).square().mean(),
                                copy_mse=target.square().mean(),
                                prediction_rms=prediction.square().mean().sqrt(),
                                target_rms=target.square().mean().sqrt())
                        item['diagnostics'] = stats
        return r, next_history, next_r, item

    @torch.no_grad()
    def gain_diagnostics(self):
        values = self.gain.detach().float().flatten()
        absolute = values.abs()
        quantiles = torch.quantile(absolute, absolute.new_tensor([0.5, 0.95, 0.99]))
        return {'patch': {
            'rms': float(values.square().mean().sqrt()),
            'abs_p50': float(quantiles[0]), 'abs_p95': float(quantiles[1]),
            'abs_p99': float(quantiles[2]), 'abs_max': float(absolute.max()),
            'fraction_abs_gt_1': float((absolute > 1).float().mean()),
            'fraction_abs_gt_2': float((absolute > 2).float().mean()),
            'effective_abs_max': float((self.alpha * values).abs().max()),
            'mean_gain_derivative': self.alpha,
        }}
