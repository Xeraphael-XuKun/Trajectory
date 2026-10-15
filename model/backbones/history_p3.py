"""P3: full-token C0 velocity plus a separately gained patch innovation."""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryP3Adapter(nn.Module):
    uses_velocity = True

    def __init__(self, dim, depth, patches, hidden_dim=32, alpha=0.25,
                 prediction_layers=None):
        super().__init__()
        self.alpha = float(alpha)  # innovation only; velocity coefficient is 1.
        self.token_scope, self.gain_mode = 'all', 'linear'
        self.prediction_layers = frozenset(range(3, depth + 1))
        if prediction_layers is not None and set(prediction_layers) != self.prediction_layers:
            raise ValueError('P3 predicts at every block from 3 through depth')
        self.register_buffer('specification', torch.tensor([103, dim, depth, patches, hidden_dim, alpha]))
        self.velocity_gain = nn.Parameter(torch.zeros(depth - 1, 1, patches + 1, dim))
        self.innovation_gain = nn.Parameter(torch.zeros(depth - 2, 1, patches, dim))
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            for _ in range(depth - 2)])
        for predictor in self.predictors:
            nn.init.zeros_(predictor[-1].weight)
            nn.init.zeros_(predictor[-1].bias)

    def correction(self, h, layer, history, previous_r, collect=False, diagnostics=False,
                   previous_velocity=None):
        with torch.autocast(device_type=h.device.type, enabled=False):
            prediction = target = None
            if layer == 0:
                # No Z1, predictor, history observation or write before block 1.
                r = torch.zeros_like(h)
                next_history = []
            else:
                z = F.layer_norm(previous_velocity.float(), (h.shape[-1],), eps=1e-5)
                velocity_r = self.velocity_gain[layer - 1] * z
                z_patch = z[:, 1:]
                if layer >= 2:
                    last = history[-1]
                    old_delta = last - history[-2] if len(history) > 1 else torch.zeros_like(last)
                    prediction = self.predictors[layer - 2](torch.cat((last, old_delta), -1))
                    innovation = z_patch - (last + prediction).detach()
                    innovation_direction = F.layer_norm(innovation, (h.shape[-1],), eps=1e-5)
                    innovation_r = self.alpha * self.innovation_gain[layer - 2] * innovation_direction
                    r = (velocity_r + torch.cat((torch.zeros_like(velocity_r[:, :1]), innovation_r), 1)).to(h.dtype)
                    if collect:
                        target = (z_patch - last).detach()
                else:
                    r = velocity_r.to(h.dtype)
                # Only the predictor's historical copy is detached, not current z/V.
                next_history = (history + [z_patch.detach()])[-2:]
            item = None
            if collect:
                item = {'prediction': prediction, 'target': target}
                if diagnostics:
                    with torch.no_grad():
                        patch_h, patch_r = h[:, 1:].float(), r[:, 1:].float()
                        cls_h, cls_r = h[:, :1].float(), r[:, :1].float()
                        stats = {
                            'correction_rms': patch_r.square().mean().sqrt(),
                            'correction_ratio': patch_r.square().mean().sqrt() / patch_h.square().mean().sqrt().clamp_min(1e-8),
                            'cls_correction_rms': cls_r.square().mean().sqrt(),
                            'cls_correction_ratio': cls_r.square().mean().sqrt() / cls_h.square().mean().sqrt().clamp_min(1e-8),
                        }
                        if layer > 0:
                            stats.update(state_variance=z_patch.var(-1, unbiased=False).mean(),
                                         velocity_rms=previous_velocity.float().square().mean().sqrt(),
                                         velocity_correction_rms=velocity_r[:, 1:].square().mean().sqrt())
                        if layer >= 2:
                            stats.update(innovation_rms=innovation.square().mean().sqrt(),
                                         normalized_innovation_rms=innovation_direction.square().mean().sqrt(),
                                         innovation_correction_rms=innovation_r.square().mean().sqrt(),
                                         prediction_mse=(prediction - target).square().mean(),
                                         copy_mse=target.square().mean(),
                                         prediction_rms=prediction.square().mean().sqrt(),
                                         target_rms=target.square().mean().sqrt())
                        item['diagnostics'] = stats
        # P3 has no Q or previous-action projection.
        return r, next_history, None, item

    @torch.no_grad()
    def gain_diagnostics(self):
        result = {}
        for name, gain, alpha in [('velocity', self.velocity_gain, 1.0),
                                  ('innovation', self.innovation_gain, self.alpha)]:
            values = gain.detach().float().flatten()
            absolute = values.abs()
            quantiles = torch.quantile(absolute, absolute.new_tensor([0.5, 0.95, 0.99]))
            result[name] = {
                'rms': float(values.square().mean().sqrt()),
                'abs_p50': float(quantiles[0]), 'abs_p95': float(quantiles[1]),
                'abs_p99': float(quantiles[2]), 'abs_max': float(absolute.max()),
                'fraction_abs_gt_1': float((absolute > 1).float().mean()),
                'fraction_abs_gt_2': float((absolute > 2).float().mean()),
                'effective_abs_max': float((alpha * values).abs().max()),
                'mean_gain_derivative': alpha,
            }
        return result
