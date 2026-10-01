"""Existing four-cell VTC view supervision (reserved, disabled in this round)."""
import torch
from .text_align import AERIAL, GROUND, text_align_loss

def view_align(aux):
    """The 2x2 view supervision.  Returns (loss, stats-for-the-log).

    Both features go through CLIP's `visual.proj` into the 512-d joint space --
    the 768-d CLS lives in the backbone's own space and is not comparable to a
    text vector at all.

    With modality anchors the anchor row is (modality, view), so every sample
    carries its own target and its own (aerial, ground) pair for the margin;
    comparing a thermal feature against the RGB anchors would be measuring the
    wrong direction.  Without them `modality` is all zeros and this reduces to
    the two-anchor case exactly.

    The `margin` entries are the point of the diagnostic: they say how far each
    group sits towards the ground anchor, so `push` (how far the delta moved
    the aerial images) and `drift` (how far it moved the ground ones, which
    should be ~0) are readable straight off the log.  A run where the delta
    never moved and a run where the idea does not work produce the same mAP;
    only these two numbers tell them apart.
    """
    proj, text, scale = aux['proj'], aux['text'], aux['logit_scale']
    is_aerial, modality = aux['is_aerial'], aux['modality']
    n_view = aux['n_view']
    to_clip = lambda f: f.float() @ proj.float()

    def rows(view_idx, sel):
        """anchor row per selected sample, for a fixed target view."""
        return modality[sel] * n_view + view_idx

    def pairs(sel):
        """[N, 2] of (aerial row, ground row) in each sample's own modality."""
        base = modality[sel] * n_view
        return torch.stack([base + AERIAL, base + GROUND], dim=1)

    groups = {
        'A_before': (aux['feat_before'], is_aerial, AERIAL),
        'A_after': (aux['feat_after'], is_aerial, GROUND),
        'G_before': (aux['feat_before'], ~is_aerial, GROUND),
        'G_after': (aux['feat_after'], ~is_aerial, GROUND),
    }
    total, stats = 0.0, {}
    for name, (feat, sel, view_idx) in groups.items():
        loss, acc, margin = text_align_loss(
            to_clip(feat[sel]), text, rows(view_idx, sel), scale, pairs(sel))
        total = total + loss
        stats['acc_' + name] = acc.item()
        stats['margin_' + name] = margin.item()
    stats['push'] = stats['margin_A_after'] - stats['margin_A_before']
    stats['drift'] = stats['margin_G_after'] - stats['margin_G_before']
    stats['n_aerial'] = int(is_aerial.sum())
    stats['n_total'] = int(is_aerial.numel())

    # Per-modality push.  With one modality supervised this is the same number
    # again; with three it answers the question the modality slot was added
    # for -- whether the correction the delta learns is shared across spectra
    # or only ever applies to the one CLIP understands.
    if aux['n_modality'] > 1:
        per = []
        for m in range(aux['n_modality']):
            sel = is_aerial & (modality == m)
            if int(sel.sum()) == 0:
                per.append(float('nan')); continue
            b = text_align_loss(to_clip(aux['feat_before'][sel]), text,
                                rows(AERIAL, sel), scale, pairs(sel))[2].item()
            a = text_align_loss(to_clip(aux['feat_after'][sel]), text,
                                rows(GROUND, sel), scale, pairs(sel))[2].item()
            per.append(a - b)
        stats['push_per_modality'] = per
    return total, stats
