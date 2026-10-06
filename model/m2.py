"""M2-A readout and M2-only configuration boundaries (no legacy method reuse)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda import amp


def validate_m2_config(cfg):
    mode = cfg.M2.MODE
    if mode == "none":
        return
    if mode not in ("a", "b"):
        raise ValueError("M2.MODE must be none, a or b (A/B are mutually exclusive)")
    if (cfg.MODEL.TRANSFORMER_TYPE != "vit_base_clip"
            or cfg.MODEL.TEXT_ALIGN or cfg.MODEL.VPR or cfg.MODEL.MOD_DELTA
            or cfg.MODEL.PE_LAYERWISE != "none" or cfg.MODEL.PE_TYPE != "learnable"
            or cfg.MODEL.PE_FREEZE_BASE or cfg.MODEL.CE_SPLIT_VIEW
            or cfg.MODEL.CE_SPLIT_MODALITY or cfg.MODEL.DIST_TRAIN
            or cfg.SOLVER.MOD_DELTA_ONLY or cfg.SOLVER.LOSS_TYPE != "base"
            or cfg.SOLVER.TEXT_LOSS_WEIGHT != 0 or cfg.SOLVER.TWIN_LOSS_WEIGHT != 0
            or cfg.SOLVER.TNCE_WEIGHT != 0):
        raise ValueError("M2 experiments require the unchanged CLIP baseline/C0, CE+Triplet only")
    if cfg.MODEL.TOKEN_TRAJECTORY and (
            cfg.MODEL.TOKEN_TRAJECTORY_VARIANT != "dense"
            or cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX != 0.0):
        raise ValueError("M2 + Trajectory requires the historical velocity-only C0")
    if (list(cfg.DATASETS.MODALITIES) != ["RGB", "IR", "Thermal"]
            or sorted(cfg.DATASETS.AERIAL_CAMS) != [5, 6]
            or cfg.DATALOADER.SAMPLER != "PKM" or cfg.TEST.METRIC != "sysu"):
        raise ValueError("M2 relations use WHU-MARS RGB/IR/Thermal, cameras 5/6, PKM, all-same-camera exclusion")
    if cfg.M2.A_OBJECTIVE not in ("none", "rank", "full"):
        raise ValueError("M2.A_OBJECTIVE must be none, rank or full")
    if cfg.M2.B_WEIGHTING not in ("uniform", "conditional"):
        raise ValueError("M2.B_WEIGHTING must be uniform or conditional")
    if (cfg.M2.WEIGHT < 0 or cfg.M2.WARMUP_EPOCHS < 0
            or cfg.M2.TEMPERATURE <= 0 or cfg.M2.B_WEIGHT_SCALE <= 0
            or cfg.M2.A_ALPHA < 0 or cfg.M2.A_HIDDEN < 1):
        raise ValueError("Invalid M2 loss/readout scale")


class M2Readout(nn.Module):
    """Single-image bounded patch residual; no modality/camera inputs."""
    def __init__(self, dim=768, hidden=64, alpha=0.1):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.alpha = float(alpha)
        self.register_buffer("alpha_setting", torch.tensor(float(alpha)))

    def forward(self, cls, patches):
        # Identical arithmetic in main and detached auxiliary passes, including AMP.
        with amp.autocast(enabled=False):
            g, p = cls.float(), patches.float()
            q = torch.cat((F.layer_norm(p, (p.shape[-1],)),
                           F.layer_norm(g, (g.shape[-1],))[:, None].expand_as(p)), -1)
            logits = self.scorer(q).squeeze(-1)
            null = (self.null_bias + math.log(p.shape[1])).expand(p.shape[0], 1)
            weights = torch.softmax(torch.cat((null, logits), 1), 1)
            centered = p - p.mean(1, keepdim=True)
            scale = centered.flatten(1).norm(dim=1, keepdim=True) / math.sqrt(p.shape[1]) + 1e-8
            residuals = centered / scale[:, :, None]
            v = (weights[:, 1:, None] * residuals).sum(1)
            bounded = v / (1.0 + (v.square().sum(-1, keepdim=True) + 1e-12).sqrt())
            correction = self.alpha * g.norm(dim=-1, keepdim=True).detach() * bounded
            result = (g + correction).to(cls.dtype)
            stats = {
                "null_weight": weights[:, 0].detach().mean(),
                "residual_ratio": (correction.norm(dim=-1) /
                                   g.norm(dim=-1).clamp_min(1e-8)).detach().mean(),
            }
        return result, stats


def frozen_bn_pair(reader, cls, patches, bn):
    """Auxiliary gradients reach only reader; BN state is read once, never updated."""
    with amp.autocast(enabled=False):
        before = cls.detach().float()
        after, _ = reader(before, patches.detach())
        # Clone prevents a later normal forward from mutating saved BN buffers.
        mean, var = bn.running_mean.detach().clone(), bn.running_var.detach().clone()
        weight = bn.weight.detach() if bn.weight is not None else None
        bias = bn.bias.detach() if bn.bias is not None else None
        def project(x):
            return F.normalize(F.batch_norm(x, mean, var, weight, bias,
                                            training=False, eps=bn.eps), dim=1)
        return project(after), project(before).detach()
