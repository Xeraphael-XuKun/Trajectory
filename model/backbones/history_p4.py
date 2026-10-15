"""P4: three-observation convex history prior and scalar patch gating of C0."""
import torch
from torch import nn
from torch.nn import functional as F


def rms(x):
    return x.square().mean(dim=-1).sqrt()


class HistoryP4Adapter(nn.Module):
    uses_velocity = True

    def __init__(self, dim, depth, patches, eta=0.5, prediction_layers=None):
        super().__init__()
        self.eta, self.stat_eps = float(eta), 1e-6
        self.token_scope, self.gain_mode = 'all', 'linear'
        self.prediction_layers = frozenset(range(4, depth + 1))
        if prediction_layers is not None and set(prediction_layers) != self.prediction_layers:
            raise ValueError('P4 prediction supervision covers blocks 4 through depth')
        self.register_buffer('specification', torch.tensor([104, dim, depth, patches, eta, self.stat_eps]))
        self.velocity_gain = nn.Parameter(torch.zeros(depth - 1, 1, patches + 1, dim))
        # Rows correspond to blocks 3..depth; columns are bias, cosine, innovation, action.
        self.gate_coefficients = nn.Parameter(torch.zeros(depth - 2, 4))
        weights = []
        for layer in range(4, depth + 1):
            prior = torch.tensor([.8, .15] if layer == 4 else [.8, .15, .05])
            weights.append(nn.Parameter((prior / prior.sum()).log()))
        self.history_logits = nn.ParameterList(weights)

    def correction(self, h, layer, history, previous_r, collect=False, diagnostics=False,
                   previous_velocity=None):
        with torch.autocast(device_type=h.device.type, enabled=False):
            prediction = target = None
            if layer == 0:
                r = torch.zeros_like(h)
                next_history = []
            else:
                z = F.layer_norm(previous_velocity.float(), (h.shape[-1],), eps=1e-5)
                z_patch = z[:, 1:]
                candidate = self.velocity_gain[layer - 1] * z
                if layer >= 2:
                    count = min(3, layer - 1)
                    if layer == 2:
                        predicted_z = history[-1]  # block 3 has no learned history weight.
                    else:
                        weights = self.history_logits[layer - 3].softmax(dim=0)
                        predicted_z = sum(weights[k] * history[-1 - k] for k in range(count))
                    # Statistics are observations only; no task gradient through this branch.
                    with torch.no_grad():
                        reference = predicted_z.detach()
                        innovation = z_patch.detach() - reference
                        cosine = (z_patch * reference).sum(-1) / (
                            z_patch.norm(dim=-1).clamp_min(self.stat_eps)
                            * reference.norm(dim=-1).clamp_min(self.stat_eps))
                        relative_innovation = torch.log((rms(innovation) + self.stat_eps)
                                                        / (rms(z_patch) + self.stat_eps)).tanh()
                        previous_action = torch.log1p(rms(previous_r)
                                                      / (rms(h[:, 1:].float()) + self.stat_eps)).tanh()
                        statistics = torch.stack((cosine, relative_innovation, previous_action), -1)
                    coefficients = self.gate_coefficients[layer - 2]
                    activation = coefficients[0] + (statistics * coefficients[1:]).sum(-1, keepdim=True)
                    multiplier = 1.0 + self.eta * activation.tanh()
                    r = torch.cat((candidate[:, :1], multiplier * candidate[:, 1:]), 1).to(h.dtype)
                    if collect and layer >= 3:
                        # Same-last-state subtraction yields the specified increment loss.
                        prediction = predicted_z - history[-1]
                        target = (z_patch - history[-1]).detach()
                else:
                    r = candidate.to(h.dtype)
                next_history = (history + [z_patch.detach()])[-3:]
            # Store the actual previous patch action, including its multiplier and dtype cast.
            next_r = r[:, 1:].detach().float()
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
                                         c0_candidate_rms=candidate[:, 1:].square().mean().sqrt())
                        if layer >= 2:
                            stats.update(innovation_rms=innovation.square().mean().sqrt(),
                                         cosine_mean=cosine.mean(), relative_innovation_mean=relative_innovation.mean(),
                                         previous_action_mean=previous_action.mean(),
                                         multiplier_mean=multiplier.mean(), multiplier_min=multiplier.min(),
                                         multiplier_max=multiplier.max())
                        if layer >= 3:
                            stats.update(prediction_mse=(prediction - target).square().mean(),
                                         copy_mse=target.square().mean(),
                                         prediction_rms=prediction.square().mean().sqrt(),
                                         target_rms=target.square().mean().sqrt())
                        item['diagnostics'] = stats
        return r, next_history, next_r, item

    @torch.no_grad()
    def history_weights_diagnostics(self):
        return {'3': [1.0], **{str(layer): logits.softmax(0).cpu().tolist()
                               for layer, logits in enumerate(self.history_logits, start=4)}}

    @torch.no_grad()
    def gain_diagnostics(self):
        values = self.velocity_gain.detach().float().flatten()
        absolute = values.abs()
        quantiles = torch.quantile(absolute, absolute.new_tensor([0.5, 0.95, 0.99]))
        return {'velocity': {
            'rms': float(values.square().mean().sqrt()),
            'abs_p50': float(quantiles[0]), 'abs_p95': float(quantiles[1]),
            'abs_p99': float(quantiles[2]), 'abs_max': float(absolute.max()),
            'fraction_abs_gt_1': float((absolute > 1).float().mean()),
            'fraction_abs_gt_2': float((absolute > 2).float().mean()),
            'effective_abs_max': float(absolute.max()), 'mean_gain_derivative': 1.0,
        }}
