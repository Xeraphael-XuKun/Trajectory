"""Label-based M2 objectives. Relation masks use actual pid/camera/modality labels."""
import torch
from torch.nn import functional as F
from torch.cuda import amp


def _cells(camera, modality, aerial_cams):
    aerial = torch.zeros_like(camera, dtype=torch.bool)
    for cam in aerial_cams:
        aerial |= camera == cam
    return aerial.long() * 3 + modality


def m2_a_loss(after, before, pid, camera, modality, cfg, aerial_cams=(5, 6)):
    with amp.autocast(enabled=False):
        z1, z0 = after.float(), before.detach().float()
        s1, s0 = z1 @ z1.t(), z0 @ z0.t()
        same_id = pid[:, None] == pid[None, :]
        different_cam = camera[:, None] != camera[None, :]
        cells = _cells(camera, modality, aerial_cams)
        zero = z1.sum() * 0.0
        ranks, keeps = [], []
        pair_count = anchor_count = 0
        gain_sum, harmed_count = zero.detach(), zero.detach()
        for cell in range(6):
            eligible = different_cam & (cells[None, :] == cell)
            pos, neg = eligible & same_id, eligible & ~same_id
            valid = pos.any(1) & neg.any(1)
            if not valid.any():
                continue
            # Hardest negative is selected by corrected similarity, reused for both.
            hardest = s1.detach().masked_fill(~neg, -1e4).argmax(1, keepdim=True)
            d1 = s1 - s1.gather(1, hardest)
            d0 = s0 - s0.gather(1, hardest)
            rank = cfg.TEMPERATURE * F.softplus((cfg.MARGIN - d1) / cfg.TEMPERATURE)
            keep = F.relu(d0 - d1 - cfg.A_KEEP_TOL)
            denom = pos.sum(1).clamp_min(1)
            ranks.append(((rank * pos).sum(1) / denom)[valid].mean())
            keeps.append(((keep * pos).sum(1) / denom)[valid].mean())
            pairs = pos & valid[:, None]
            pair_count += int(pairs.sum())
            anchor_count += int(valid.sum())
            gain_sum = gain_sum + (d1 - d0).detach()[pairs].sum()
            harmed_count = harmed_count + ((d0 - d1).detach()[pairs] > cfg.A_KEEP_TOL).sum()
        rank = torch.stack(ranks).mean() if ranks else zero
        keep = torch.stack(keeps).mean() if keeps else zero
        rho = 1.0 if cfg.A_OBJECTIVE == "full" else 0.0
        loss = rank + rho * keep
        stats = dict(valid_cells=len(ranks), anchor_conditions=anchor_count,
                     positive_pairs=pair_count, rank=float(rank.detach()),
                     keep=float(keep.detach()), mean_gap_gain=float(gain_sum) / max(1, pair_count),
                     harmed_fraction=float(harmed_count) / max(1, pair_count))
        return loss, stats


def m2_b_from_gaps(gaps, valid, cfg):
    """gaps: [anchor, wrong-id, platform, modality]; valid: first three axes."""
    selected = gaps[valid]
    if selected.numel() == 0:
        return gaps.sum() * 0.0, dict(valid_relations=0, active_terms=0,
                                     weight_mean=0.0, ordinary_loss=0.0, gap_mean=0.0)
    other_max = torch.stack([selected[:, [n for n in range(3) if n != m]].max(1)[0]
                             for m in range(3)], 1)
    if cfg.B_WEIGHTING == "uniform":
        weights = torch.ones_like(selected)
    else:
        weights = ((selected - other_max - cfg.B_DELTA).relu() /
                   cfg.B_WEIGHT_SCALE).clamp(0.0, 1.0).detach()
    penalty = (selected + cfg.MARGIN).relu().square()
    loss = (weights * penalty).mean()  # includes inactive terms in the denominator
    stats = dict(valid_relations=selected.shape[0],
                 active_terms=int(((weights > 0) & (penalty > 0)).sum()),
                 weight_mean=float(weights.detach().mean()),
                 ordinary_loss=float(penalty.detach().mean()),
                 gap_mean=float(selected.detach().mean()))
    for m in range(3):
        stats["active_mod_{}".format(m)] = int(((weights[:, m] > 0) & (penalty[:, m] > 0)).sum())
    return loss, stats


def m2_b_loss(feat, pid, camera, modality, cfg, aerial_cams=(5, 6)):
    with amp.autocast(enabled=False):
        z = F.normalize(feat.float(), dim=1)
        similarity = z @ z.t()
        identities, own_id = torch.unique(pid, sorted=True, return_inverse=True)
        different_cam = camera[:, None] != camera[None, :]
        identity_mask = identities[:, None] == pid[None, :]
        cells = _cells(camera, modality, aerial_cams)
        scores, exists = [], []
        # Six [B,K,B] masks; avoid thousands of Python anchor/identity loops.
        for cell in range(6):
            mask = (different_cam[:, None, :] & identity_mask[None, :, :]
                    & (cells[None, None, :] == cell))
            count = mask.sum(-1)
            logits = (similarity[:, None, :] / cfg.TEMPERATURE).expand_as(mask)
            score = cfg.TEMPERATURE * (
                logits.masked_fill(~mask, -1e4).logsumexp(-1) -
                count.clamp_min(1).float().log())
            scores.append(score)
            exists.append(count > 0)
        scores = torch.stack(scores, -1).reshape(len(pid), len(identities), 2, 3)
        exists = torch.stack(exists, -1).reshape_as(scores)
        rows = torch.arange(len(pid), device=pid.device)
        positive = scores[rows, own_id]
        pos_exists = exists[rows, own_id].all(-1)
        gaps = scores - positive[:, None]
        wrong = identities[None, :] != pid[:, None]
        valid = exists.all(-1) & pos_exists[:, None, :] & wrong[:, :, None]
        return m2_b_from_gaps(gaps, valid, cfg)
