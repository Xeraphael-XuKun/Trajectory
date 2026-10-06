"""Condition-matched identity margins for C0 probes, independent of legacy losses."""
import torch
import torch.nn.functional as F


def condition_masks(ids, cams, mods, aerial_cams=(5, 6)):
    aerial = (cams[:, None] == cams.new_tensor(aerial_cams)[None, :]).any(1)
    same_id = ids[:, None] == ids[None, :]
    cross_cam = cams[:, None] != cams[None, :]
    same_mod = mods[:, None] == mods[None, :]
    same_view = aerial[:, None] == aerial[None, :]
    groups = (same_mod & same_view, ~same_mod & same_view,
              same_mod & ~same_view, ~same_mod & ~same_view)
    return same_id, cross_cam, same_mod & same_view, groups


def layer_loss(reference, corrected, ids, cams, mods, settings, variant):
    # FP32 similarity matrices even under the outer CLIP autocast context.
    with torch.cuda.amp.autocast(enabled=False):
        s0 = reference.detach().float() @ reference.detach().float().t()
        s1 = corrected.float() @ corrected.float().t()
        same_id, cross_cam, same_cell, groups = condition_masks(ids, cams, mods)
        negative = ~same_id & cross_cam
        # [anchor, positive, candidate negative]: fixed metadata pool per pair.
        candidates = negative[:, None, :] & same_cell[None, :, :]
        valid_negative = candidates.any(2)
        with torch.no_grad():
            nearest = s0[:, None, :].expand_as(candidates).masked_fill(
                ~candidates, -float('inf')).argmax(2)
        d0 = (s0 - s0.gather(1, nearest)).detach()
        d1 = s1 - s1.gather(1, nearest)
        target = (settings.MARGIN - d0).clamp(min=0, max=settings.MAX_GAIN)
        cross_loss = (F.relu(settings.MARGIN - d1) if variant == 'c1_absolute'
                      else F.relu(d0 + target - d1))
        keep_loss = F.relu(d0 - settings.KEEP_TOL - d1)
        losses, weights, stats = [], [], []
        for index, group in enumerate(groups):
            valid = same_id & cross_cam & group & valid_negative
            counts = valid.sum(1)
            anchors = counts > 0
            values = keep_loss if index == 0 else cross_loss
            per_anchor = (values * valid).sum(1) / counts.clamp_min(1)
            weight = anchors.sum()
            loss = per_anchor.sum() / weight.clamp_min(1)
            losses.append(loss)
            weights.append(weight)
            delta = (d1 - d0).detach()[valid]
            # Stats stay on-device until the caller transfers one small table.
            if delta.numel():
                row = torch.stack((weight.float(), valid.sum().float(),
                    d0[valid].sum(), d1.detach()[valid].sum(), delta.sum(),
                    delta.square().sum(), (delta < 0).sum().float(),
                    (values.detach()[valid] > 0).sum().float(),
                    delta.min(), delta.max(), loss.detach()))
            else:
                row = d0.new_zeros(11)
            stats.append(row)
        cross_weights = torch.stack(weights[1:]).float()
        if variant != 'c1_pooled':
            cross_weights = (cross_weights > 0).float()
        cross = (torch.stack(losses[1:]) * cross_weights).sum() / cross_weights.sum().clamp_min(1)
        return cross + settings.KEEP_WEIGHT * losses[0], torch.stack(stats)


def c0_aux_loss(aux, ids, settings):
    losses, statistics = [], {}
    for layer, (reference, corrected, ratio) in aux['probes'].items():
        loss, rows = layer_loss(reference, corrected, ids, aux['cams'], aux['mods'],
                                settings, settings.VARIANT)
        losses.append(loss)
        statistics[layer + 1] = (rows.detach(), ratio.detach())
    # All3 averages all three full layer losses (including the keep term).
    return torch.stack(losses).mean(), statistics
