"""P3-A: learnable per-channel baseline for the original P3 history predictor."""
import torch
from torch import nn
from .history_p3 import HistoryP3Adapter


class HistoryP3AAdapter(HistoryP3Adapter):
    def __init__(self, dim, depth, patches, hidden_dim=32, alpha=0.25,
                 prediction_layers=None):
        super().__init__(dim, depth, patches, hidden_dim, alpha, prediction_layers)
        # One vector per prediction block (3..depth); shared across images/patches.
        self.history_retention = nn.Parameter(torch.ones(depth - 2, 1, 1, dim))
        self.specification[0] = 113

    def _predict_history(self, index, last, old_delta):
        nonlinear = self.predictors[index](torch.cat((last, old_delta), -1))
        retention = self.history_retention[index]
        # Supervise the complete increment, including the learnable history path.
        delta = (retention - 1.0) * last + nonlinear
        predicted_state = retention * last + nonlinear
        return delta, predicted_state

    @torch.no_grad()
    def history_retention_diagnostics(self):
        values = self.history_retention.detach().float().flatten(1)
        return {str(layer): {
            'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
            'rms_from_one': float((v - 1).square().mean().sqrt()),
            'fraction_negative': float((v < 0).float().mean()),
        } for layer, v in enumerate(values, start=3)}
