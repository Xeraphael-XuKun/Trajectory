"""P3-B: innovation normalization relative to current normalized-velocity scale."""
import torch
from .history_p3 import HistoryP3Adapter


class HistoryP3BAdapter(HistoryP3Adapter):
    def __init__(self, dim, depth, patches, hidden_dim=32, alpha=0.25,
                 prediction_layers=None):
        super().__init__(dim, depth, patches, hidden_dim, alpha, prediction_layers)
        self.tau, self.epsilon = 0.25, 1e-5
        self.specification = torch.cat((self.specification,
                                        self.specification.new_tensor([self.tau, self.epsilon])))
        self.specification[0] = 114

    def _normalize_innovation(self, innovation, z_patch):
        # Called inside P3's FP32 branch. Only the reference variance is detached.
        centered = innovation - innovation.mean(dim=-1, keepdim=True)
        innovation_variance = centered.square().mean(dim=-1, keepdim=True)
        reference_variance = z_patch.var(dim=-1, unbiased=False, keepdim=True).detach()
        denominator = (innovation_variance + self.tau ** 2 * reference_variance
                       + self.epsilon).sqrt()
        return centered / denominator
