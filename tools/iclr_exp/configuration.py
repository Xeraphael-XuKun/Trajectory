"""Actual ordered YACS merge and constraints defining this experiment."""
import os
from pathlib import Path
from config import cfg as defaults
from yacs.config import CfgNode

ROOT = Path(__file__).resolve().parents[2]


def load_config(base=None, common=None, variant=None, resources=None, opts=()):
    c = defaults.clone()
    for path in (base or ROOT / 'configs/trajectory_C0.yml',
                 common or ROOT / 'configs/iclr_exp/common.yml',
                 variant or ROOT / 'configs/iclr_exp/tgva_ref.yml', resources):
        if path:
            with open(path, encoding='utf-8') as stream:
                c.merge_from_other_cfg(CfgNode.load_cfg(stream))
    c.merge_from_list(list(opts))
    validate(c)
    c.freeze()
    return c


def validate(c):
    if not c.ICLR_EXP.ENABLED:
        return
    checks = {
        'only implemented variant tgva_ref': c.ICLR_EXP.VARIANT == 'tgva_ref',
        'independent DropPath / trainable projection': c.ICLR_EXP.PAIR_RNG == 'independent' and c.ICLR_EXP.PROJECTION_TRAINABLE,
        'original dense velocity-only C0': c.MODEL.TOKEN_TRAJECTORY and c.MODEL.TOKEN_TRAJECTORY_VARIANT == 'dense' and c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX == 0.,
        'CLIP ViT / original pretrained initialization': c.MODEL.TRANSFORMER_TYPE == 'vit_base_clip' and c.MODEL.PRETRAIN_CHOICE == 'imagenet',
        'legacy auxiliaries disabled': not any((c.M2.ENABLED, c.MODEL.TEXT_ALIGN, c.MODEL.VPR, c.MODEL.MOD_DELTA, c.MODEL.SIE_CAMERA, c.MODEL.SIE_VIEW, c.MODEL.CE_SPLIT_VIEW, c.MODEL.CE_SPLIT_MODALITY)),
        'original positional encoding': c.MODEL.PE_TYPE == 'learnable' and c.MODEL.PE_LAYERWISE == 'none' and not c.MODEL.PE_FREEZE_BASE,
        'original image geometry/augmentation': list(c.INPUT.SIZE_TRAIN)==[256,128] and list(c.INPUT.SIZE_TEST)==[256,128] and list(c.MODEL.STRIDE_SIZE)==[16,16] and c.INPUT.PROB==.5 and c.INPUT.RE_PROB==.5 and c.INPUT.PADDING==10 and list(c.INPUT.PIXEL_MEAN)==[.48145466,.4578275,.40821073] and list(c.INPUT.PIXEL_STD)==[.26862954,.26130258,.27577711] and c.MODEL.DROP_PATH==.1 and not c.MODEL.LAYER_SCALE and not c.MODEL.CHART_INPUT_NORM,
        'unsynchronized PKM': c.DATALOADER.SAMPLER == 'PKM' and not c.DATALOADER.SYNC_FRAMES and not c.DATALOADER.TIE_AUGMENTATION and not c.DATALOADER.TIE_ERASING,
        'three modalities / Ground=0 Aerial=1': list(c.DATASETS.MODALITIES) == ['RGB', 'IR', 'Thermal'] and list(c.DATASETS.AERIAL_CAMS) == [5, 6],
        'unmodified identity objective': c.SOLVER.LOSS_TYPE == 'base' and c.MODEL.IF_LABELSMOOTH == 'off' and c.MODEL.IF_WITH_CENTER == 'no' and c.MODEL.METRIC_LOSS_TYPE == 'triplet' and c.MODEL.ID_LOSS_WEIGHT == 1. and c.MODEL.TRIPLET_LOSS_WEIGHT == 1.,
        'no extra auxiliary losses': all(getattr(c.SOLVER, k) == 0. for k in ('TEXT_LOSS_WEIGHT', 'TWIN_LOSS_WEIGHT', 'TNCE_WEIGHT', 'VPR_VIEW_TEXT_WEIGHT', 'VPR_CSD_WEIGHT')) and not c.SOLVER.MOD_DELTA_ONLY,
        'reference TGVA settings': c.ICLR_EXP.TGVA.WEIGHT == 5. and c.ICLR_EXP.TGVA.N_CONTEXT == 4 and c.ICLR_EXP.TGVA.CONTEXT_INIT_STD == .02 and list(c.ICLR_EXP.TGVA.MODALITY_WORDS) == ['rgb', 'infrared', 'thermal'] and list(c.ICLR_EXP.TGVA.VIEW_WORDS) == ['ground', 'aerial'] and c.ICLR_EXP.TGVA.SCALE_SOURCE == 'original_clip_logit_scale' and c.ICLR_EXP.TGVA.ANCHOR_ORDER == '2*modality+platform',
        'fixed recipe': c.SOLVER.MAX_EPOCHS == 60 and c.SOLVER.SEED == 1234 and c.SOLVER.OPTIMIZER_NAME == 'Adam' and c.SOLVER.IMS_PER_BATCH == 64 and c.DATALOADER.NUM_INSTANCE == 4 and c.SOLVER.BASE_LR == .00035 and c.SOLVER.PRETRAINED_LR == .000005 and c.SOLVER.WEIGHT_DECAY == .0001 and c.SOLVER.WEIGHT_DECAY_BIAS == .0001 and c.SOLVER.WARMUP_ITERS == 100 and c.SOLVER.LR_MIN_FACTOR == .002 and c.SOLVER.WARMUP_LR_FACTOR == .01 and c.SOLVER.BIAS_LR_FACTOR == 1 and not c.SOLVER.LARGE_FC_LR and c.SOLVER.PE_DELTA_LR_MULT == 1.,
        'single GPU': not c.MODEL.DIST_TRAIN and int(os.environ.get('WORLD_SIZE', '1')) == 1,
        'fixed observation periods': c.SOLVER.EVAL_PERIOD==10 and c.SOLVER.CHECKPOINT_PERIOD==60 and c.SOLVER.LOG_PERIOD==50,
        'ALL post-BN protocol': c.DATASETS.PROTOCOL == 'ALL' and c.TEST.NECK_FEAT == 'after' and c.TEST.FEAT_NORM == 'yes' and c.TEST.METRIC == 'sysu' and not c.TEST.RE_RANKING and c.TEST.TOP_K_EVAL == 0,
        'one original CLIP source': bool(c.MODEL.PRETRAIN_PATH) and os.path.abspath(c.MODEL.PRETRAIN_PATH) == os.path.abspath(c.ICLR_EXP.ASSETS.ORIGINAL_CLIP),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError('I1 definition mismatch: ' + '; '.join(failed))


def make_optimizer(c, model, center):
    import torch
    groups, roles = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            roles.append(dict(name=name,role='frozen',numel=parameter.numel(),lr=None,
                              weight_decay=None,requires_grad=False))
            continue
        if name == 'base.clip_proj':
            role, lr = 'visual_projection', c.SOLVER.PRETRAINED_LR
        elif name.startswith('base.token_trajectory.'):
            role, lr = 'c0_gain', c.SOLVER.BASE_LR
        elif name.startswith('base.'):
            role, lr = 'pretrained_visual', c.SOLVER.PRETRAINED_LR
        elif name.startswith('iclr_tgva.prompts.'):
            role, lr = 'condition_prompt', c.SOLVER.BASE_LR
        elif name.startswith(('bottleneck.', 'classifier.')):
            role, lr = 'identity_head', c.SOLVER.BASE_LR
        else:
            raise ValueError('unassigned I1 optimizer parameter: ' + name)
        groups.append(dict(params=[parameter], lr=lr, weight_decay=.0001))
        roles.append(dict(name=name, role=role, numel=parameter.numel(),
                          lr=lr, weight_decay=.0001, requires_grad=True))
    model.iclr_optimizer_roles = roles
    return torch.optim.Adam(groups), torch.optim.SGD(center.parameters(), lr=c.SOLVER.CENTER_LR)
