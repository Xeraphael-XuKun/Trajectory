import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        decay = cfg.SOLVER.WEIGHT_DECAY
        if key.startswith('base.') and '.token_trajectory.' not in key:
            lr = cfg.SOLVER.PRETRAINED_LR
        if 'bias' in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if '.token_trajectory.' in key:
            lr *= cfg.SOLVER.PE_DELTA_LR_MULT
        params.append({'params': [value], 'lr': lr, 'weight_decay': decay})
    return getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
