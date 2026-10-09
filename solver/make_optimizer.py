import torch


def make_optimizer(cfg, model, center_criterion):
    if cfg.SOLVER.MOD_DELTA_ONLY:
        # v1's central claim, made mechanical rather than argued: with every
        # other parameter frozen, an evaluation with TEST.MOD_DELTA False is the
        # starting checkpoint bit for bit, so any cross-modality movement is
        # attributable to the banks and nothing else.  Done here rather than in
        # the model so the freeze and the parameter list cannot disagree.
        if not cfg.MODEL.MOD_DELTA:
            raise ValueError('SOLVER.MOD_DELTA_ONLY needs MODEL.MOD_DELTA True')
        kept = 0
        for key, value in model.named_parameters():
            if 'mod_delta' in key:
                kept += value.numel()
            else:
                value.requires_grad_(False)
        frozen = sum(v.numel() for v in model.parameters()) - kept
        if kept == 0:
            raise ValueError('SOLVER.MOD_DELTA_ONLY froze everything: no '
                             '`mod_delta` parameter exists on this model')
        print('MOD_DELTA_ONLY: frozen {:,} params, trainable {:,} (mod_delta only)'
              .format(frozen, kept))

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if (cfg.SOLVER.PRETRAINED_LR > 0 and key.startswith('base.')
                and 'pos_delta' not in key and 'mod_delta' not in key
                and '.vpr.' not in key
                and '.token_trajectory.' not in key
                and 'rope.alpha' not in key):
            # HiHR section 4.2: 3.5e-4 for randomly initialised modules, 5e-6
            # for pretrained components.  `base` is the pretrained tower;
            # classifier and bottleneck are built fresh here and keep BASE_LR.
            # pos_delta and mod_delta live inside `base` but start at zero, so
            # they count as new modules, not as pretrained weights.  A
            # zero-initialised bank on the backbone's 5e-6 clock would barely
            # leave the floor.  `rope.alpha` is the same story in one scalar per
            # layer: it starts at zero and has to travel far enough to turn the
            # rotation on at all, so on the 5e-6 clock the arm would report "the
            # model does not want rotation" when the truth is that alpha never
            # got the chance to move.
            lr = cfg.SOLVER.PRETRAINED_LR
        if 'deep_c0_controller' in key:
            # M2-3 context/projections are new parameters despite living under
            # the pretrained visual tower; keep the prescribed 3.5e-4 clock.
            lr = cfg.SOLVER.BASE_LR
        if "bias" in key:
            # Multiplies whatever lr the branch above chose.  Identical to the
            # old `BASE_LR * factor` whenever PRETRAINED_LR is off.
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        if "rope.alpha" in key:
            # A gain on the phase, not a weight.  Decaying it toward zero pulls
            # the arm back to the baseline, which is exactly the null result
            # this experiment is trying to distinguish from a real one.
            weight_decay = 0.0
        if "rope.freqs" in key:
            # A frequency is a coordinate scale, not a weight: decaying it
            # toward zero shrinks every rotation angle, and in the limit RoPE
            # collapses to the identity. Only reachable with
            # MODEL.ROPE_FREQ_TRAINABLE True -- frozen frequencies never get
            # here, the requires_grad check above already skipped them.
            weight_decay = 0.0
        if ("pos_delta" in key or ".vpr." in key
                or ".token_trajectory." in key
                or "rope.alpha" in key):
            # Zero-initialised while everything around it is pretrained, so it
            # may need a faster clock than the backbone to get moving at all.
            # The rotary gain shares the multiplier: it is the same actuator
            # role (a zero-initialised increment on the position path) and the
            # 0.5 / 1.0 / 2.0 sweep on the additive form already settled that
            # 1.0 sits on the safe side of the curve.
            lr = lr * cfg.SOLVER.PE_DELTA_LR_MULT
        if "chart_generator" in key:
            # Applied on top of the branches above, so a chart bias ends up at
            # BASE_LR * BIAS_LR_FACTOR * CHART_LR_MULT.
            lr = lr * cfg.SOLVER.CHART_LR_MULT

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
