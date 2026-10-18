"""Disjoint task/prediction clipping for the second-round History candidates."""
import torch


def history_gradient_groups(model):
    model = model.module if hasattr(model, 'module') else model
    prediction = list(model.base.history_adapter.predictors.parameters())
    prediction_ids = {id(p) for p in prediction}
    task = [p for p in model.parameters() if p.requires_grad and id(p) not in prediction_ids]
    return {'task': task, 'prediction': prediction}


def clip_history_gradients(groups, max_norm=1.0):
    # GradScaler.unscale_(optimizer) must have run before this call.
    return {name: torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
            for name, parameters in groups.items()}
