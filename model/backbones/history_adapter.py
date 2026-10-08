"""History-conditioned innovation adaptation; PDF pp. 5, 8-9, 13-14.

History is local to one image forward. This replaces, rather than stacks with,
Trajectory. The original ViT block itself is never reimplemented here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryInnovationAdapter(nn.Module):
    def __init__(self, dim, depth, patches, rank=64, alpha=0.25,
                 prediction_layers=None, token_scope='patch', gain_mode='tanh'):
        super().__init__()
        if not 0 < rank <= dim or depth < 2 or alpha <= 0:
            raise ValueError('Invalid history rank, depth or alpha')
        if token_scope not in ('patch', 'all') or gain_mode not in ('tanh', 'linear'):
            raise ValueError('Unknown history token scope or gain mode')
        self.token_scope = token_scope
        self.gain_mode = gain_mode
        self.depth = depth
        self.alpha = float(alpha)
        # Public layer numbers are 1-based; layer 1 has no prediction target.
        layers = list(range(2, depth + 1)) if prediction_layers is None else list(prediction_layers)
        if not layers or len(set(layers)) != len(layers) or any(l < 2 or l > depth for l in layers):
            raise ValueError('PREDICTION_LAYERS must be distinct layers in [2, depth]')
        self.prediction_layers = frozenset(layers)
        q, _ = torch.linalg.qr(torch.randn(dim, rank), mode='reduced')
        self.register_buffer('coordinate', q)
        # Detect wrong inference alpha / shape on disk reload, not just tensor sizes.
        # Preserve the original HI0/HI1 signature; other formulas require new metadata.
        spec = [1, dim, depth, patches, rank, alpha]
        if token_scope != 'patch' or gain_mode != 'tanh':
            spec = [2, dim, depth, patches, rank, alpha,
                    int(token_scope == 'all'), int(gain_mode == 'linear')]
        self.register_buffer('specification', torch.tensor(spec))
        self.gain = nn.Parameter(torch.zeros(depth, 1, patches, dim))
        # Extra zero rows consume no RNG; shared predictor / direction weights stay matched.
        self.cls_gain = nn.Parameter(torch.zeros(depth, 1, 1, dim)) if token_scope == 'all' else None
        # No unused predictor for the first layer.
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * rank, rank), nn.GELU(), nn.Linear(rank, rank))
            for _ in range(depth - 1)])
        self.anchors = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(depth)])
        self.correctors = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * rank, rank), nn.GELU(), nn.Linear(rank, dim))
            for _ in range(depth)])


    def _token_update(self, tokens, gain, layer, history, previous_r):
        s = F.layer_norm(tokens, (tokens.shape[-1],)) @ self.coordinate
        prediction = target = None
        if history:
            last = history[-1]
            old_delta = last - history[-2] if len(history) > 1 else torch.zeros_like(last)
            prediction = self.predictors[layer - 1](torch.cat((last, old_delta), -1))
            innovation = s - (last + prediction)
            target = (s - last).detach()
        else:
            innovation = s
        previous = torch.zeros_like(s) if previous_r is None else previous_r
        direction = self.anchors[layer](innovation)
        direction = direction + self.correctors[layer](torch.cat((s, innovation, previous), -1))
        effective_gain = gain.tanh() if self.gain_mode == 'tanh' else gain
        residual = self.alpha * effective_gain * direction
        return residual, s, innovation, prediction, target

    def correction(self, h, layer, history, previous_r, collect=False, diagnostics=False):
        with torch.autocast(device_type=h.device.type, enabled=False):
            patches = h[:, 1:].float()
            if patches.shape[1:] != self.gain.shape[2:]:
                raise ValueError('History adapter requires its configured patch grid')
            all_tokens = self.token_scope == 'all'
            patch_history = [v[:, 1:] for v in history] if all_tokens else history
            patch_previous = previous_r[:, 1:] if all_tokens and previous_r is not None else previous_r
            patch_r, s, innovation, prediction, target = self._token_update(
                patches, self.gain[layer], layer, patch_history, patch_previous)
            if all_tokens:
                # Independent CLS history, same per-token networks. Never pool patch updates.
                cls_previous = None if previous_r is None else previous_r[:, :1]
                cls_r, cls_s, _, _, _ = self._token_update(
                    h[:, :1].float(), self.cls_gain[layer], layer,
                    [v[:, :1] for v in history], cls_previous)
                r = torch.cat((cls_r.to(h.dtype), patch_r.to(h.dtype)), 1)
                current_state = torch.cat((cls_s, s), 1)
                next_r = r.detach().float() @ self.coordinate
            else:
                r = torch.cat((torch.zeros_like(h[:, :1]), patch_r.to(h.dtype)), 1)
                current_state = s
                next_r = r[:, 1:].detach().float() @ self.coordinate
            next_history = (history + [current_state.detach()])[-2:]
            item = None
            if collect:
                # ALL variants supervise patches only, with identical loss denominator.
                item = {'prediction': prediction, 'target': target}
                if diagnostics:
                    with torch.no_grad():
                        stats = {
                            'state_variance': s.var(-1, unbiased=False).mean(),
                            'innovation_rms': innovation.square().mean().sqrt(),
                            'correction_rms': patch_r.square().mean().sqrt(),
                            'correction_ratio': patch_r.square().mean().sqrt() / patches.square().mean().sqrt().clamp_min(1e-8),
                        }
                        if all_tokens:
                            stats.update(cls_correction_rms=cls_r.square().mean().sqrt(),
                                         cls_correction_ratio=cls_r.square().mean().sqrt() / h[:, :1].float().square().mean().sqrt().clamp_min(1e-8))
                        if prediction is not None:
                            stats.update(prediction_mse=(prediction - target).square().mean(),
                                         copy_mse=target.square().mean(),
                                         prediction_rms=prediction.square().mean().sqrt(),
                                         target_rms=target.square().mean().sqrt())
                        item['diagnostics'] = stats
        return r, next_history, next_r, item

    @torch.no_grad()
    def gain_diagnostics(self):
        result = {}
        for name, gain in [('patch', self.gain), ('cls', self.cls_gain)]:
            if gain is None:
                continue
            values = gain.detach().float().flatten()
            absolute = values.abs()
            quantiles = torch.quantile(absolute, absolute.new_tensor([0.5, 0.95, 0.99]))
            effective = self.alpha * (values.tanh() if self.gain_mode == 'tanh' else values)
            result[name] = {
                'rms': float(values.square().mean().sqrt()),
                'abs_p50': float(quantiles[0]), 'abs_p95': float(quantiles[1]),
                'abs_p99': float(quantiles[2]), 'abs_max': float(absolute.max()),
                'fraction_abs_gt_1': float((absolute > 1).float().mean()),
                'fraction_abs_gt_2': float((absolute > 2).float().mean()),
                'effective_abs_max': float(effective.abs().max()),
                'mean_gain_derivative': float(self.alpha * (1 - values.tanh().square()).mean()) if self.gain_mode == 'tanh' else self.alpha,
            }
        return result
