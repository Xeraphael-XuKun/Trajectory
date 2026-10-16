"""P3-C: per-layer modality/platform offsets inside the shared predictor."""
import torch
from torch import nn
from .history_p3 import HistoryP3Adapter


class HistoryP3CAdapter(HistoryP3Adapter):
    uses_conditions = True

    def __init__(self, dim, depth, patches, hidden_dim=32, alpha=0.25,
                 prediction_layers=None):
        super().__init__(dim, depth, patches, hidden_dim, alpha, prediction_layers)
        # Zero allocation consumes no RNG; original W1/b1/W2/b2 are retained.
        self.modal_condition = nn.Parameter(torch.zeros(depth - 2, 3, hidden_dim))
        self.platform_condition = nn.Parameter(torch.zeros(depth - 2, 2, hidden_dim))
        self.specification = torch.cat((self.specification, torch.tensor([3., 2., 5., 6.])))
        self.specification[0] = 115

    def _predict_history(self, index, last, old_delta, conditions=None):
        if conditions is None:
            raise ValueError('P3-C requires per-image modality and platform conditions')
        modality, platform = conditions
        predictor = self.predictors[index]
        hidden = predictor[0](torch.cat((last, old_delta), -1))
        hidden = (hidden + self.modal_condition[index, modality][:, None, :]
                  + self.platform_condition[index, platform][:, None, :])
        return predictor[2](predictor[1](hidden))
