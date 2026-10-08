"""Detached-target increment prediction loss, PDF Eq. (20)."""
import torch
from torch.nn import functional as F


def history_prediction_loss(aux):
    terms = [F.smooth_l1_loss(aux['layers'][l]['prediction'].float(),
                            aux['layers'][l]['target'].float(), reduction='mean', beta=1.0)
             for l in aux['prediction_layers']]
    return torch.stack(terms).mean()


@torch.no_grad()
def history_diagnostics(aux):
    # Called only on logged batches. PG is a ratio of batch MSEs, not retrieval.
    result = {}
    for layer, item in aux['layers'].items():
        values = item.get('diagnostics', {})
        result[str(layer)] = {k: float(v) for k, v in values.items()}
        if 'copy_mse' in values:
            result[str(layer)]['prediction_gain'] = float(
                1.0 - values['prediction_mse'] / values['copy_mse'].clamp_min(1e-8))
    return result
