"""History-conditioned innovation adaptation; PDF pp. 5, 8-9, 13-14.

History is local to one image forward. This replaces, rather than stacks with,
Trajectory. The original ViT block itself is never reimplemented here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryInnovationAdapter(nn.Module):
    def __init__(self, dim, depth, patches, rank=64, alpha=0.25,
                 prediction_layers=None):
        super().__init__()
        if not 0 < rank <= dim or depth < 2 or alpha <= 0:
            raise ValueError('Invalid history rank, depth or alpha')
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
        self.register_buffer('specification', torch.tensor([1, dim, depth, patches, rank, alpha]))
        self.gain = nn.Parameter(torch.zeros(depth, 1, patches, dim))
        # No unused predictor for the first layer.
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * rank, rank), nn.GELU(), nn.Linear(rank, rank))
            for _ in range(depth - 1)])
        self.anchors = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(depth)])
        self.correctors = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * rank, rank), nn.GELU(), nn.Linear(rank, dim))
            for _ in range(depth)])

    def correction(self, h, layer, history, previous_r, collect=False, diagnostics=False):
        # Keep the small prediction path and residual arithmetic in FP32 under AMP.
        # Original block dtype / autocast and DropPath draws are untouched.
        with torch.autocast(device_type=h.device.type, enabled=False):
            patches = h[:, 1:].float()
            if patches.shape[1:] != self.gain.shape[2:]:
                raise ValueError('History adapter requires its configured patch grid')
            s = F.layer_norm(patches, (patches.shape[-1],)) @ self.coordinate
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
            patch_r = self.alpha * self.gain[layer].tanh() * direction
            r = torch.cat((torch.zeros_like(h[:, :1]), patch_r.to(h.dtype)), 1)
            # Cache the actually written residual, without feature normalisation.
            next_r = r[:, 1:].detach().float() @ self.coordinate
            next_history = (history + [s.detach()])[-2:]
            item = None
            if collect:
                item = {'prediction': prediction, 'target': target}
                if diagnostics:
                    with torch.no_grad():
                        stats = {
                            'state_variance': s.var(-1, unbiased=False).mean(),
                            'innovation_rms': innovation.square().mean().sqrt(),
                            'correction_rms': patch_r.square().mean().sqrt(),
                            'correction_ratio': patch_r.square().mean().sqrt() / patches.square().mean().sqrt().clamp_min(1e-8),
                        }
                        if prediction is not None:
                            stats.update(prediction_mse=(prediction - target).square().mean(),
                                         copy_mse=target.square().mean(),
                                         prediction_rms=prediction.square().mean().sqrt(),
                                         target_rms=target.square().mean().sqrt())
                        item['diagnostics'] = stats
        return r, next_history, next_r, item
