"""Terminal identity ranking repair; not enabled by historical training paths."""
import torch
import torch.nn.functional as F


def terminal_repair(reference, corrected, identities, cameras, modalities,
                    weighting='reference', temperature=0.07, aerial_cams=(5, 6)):
    """Inputs are normalized features. Weight selection never receives gradients."""
    if weighting not in ('reference', 'uniform', 'self') or temperature <= 0:
        raise ValueError('Invalid terminal repair setting')
    # Disable outer AMP for cosine scores and small ranking differences.
    with torch.cuda.amp.autocast(enabled=False):
        on = corrected.float() @ corrected.float().t()
        ref = reference.detach().float() @ reference.detach().float().t()
        same = identities[:, None].eq(identities[None, :])
        cross_cam = cameras[:, None].ne(cameras[None, :])
        aerial = torch.zeros_like(cameras, dtype=torch.bool)
        for cam in aerial_cams:
            aerial |= cameras.eq(cam)
        cross = modalities[:, None].ne(modalities[None, :]) | aerial[:, None].ne(aerial[None, :])
        positive = same & cross_cam & cross
        negative = ~same & cross_cam
        valid = positive.any(1) & negative.any(1)
        anchor, pos = (positive & valid[:, None]).nonzero(as_tuple=True)
        if anchor.numel() == 0:
            return corrected.sum() * 0, {'valid_anchors': 0, 'positive_pairs': 0}
        neg = negative[anchor]
        counts = neg.sum(1)
        differences = (on[anchor] - on[anchor, pos, None]) / temperature
        pair_loss = (F.softplus(differences) * neg).sum(1) / counts
        with torch.no_grad():
            scores = on.detach() if weighting == 'self' else ref
            hardness = (torch.sigmoid((scores[anchor] - scores[anchor, pos, None]) / temperature)
                        * neg).sum(1) / counts
            if weighting == 'uniform':
                hardness = torch.ones_like(hardness)
            sums = hardness.new_zeros(len(identities)).scatter_add_(0, anchor, hardness)
            weights = hardness / sums[anchor].clamp_min(torch.finfo(hardness.dtype).tiny)
            npairs = torch.bincount(anchor, minlength=len(identities)).float()
            weight_sq = weights.new_zeros(len(identities)).scatter_add_(0, anchor, weights.square())
            ess_fraction = (1 / weight_sq[valid]) / npairs[valid]
            off_error = (torch.sigmoid((ref[anchor] - ref[anchor, pos, None]) / temperature)
                         * neg).sum(1) / counts
            on_error = (torch.sigmoid(differences.detach()) * neg).sum(1) / counts
            stats = {'valid_anchors': int(valid.sum()), 'positive_pairs': int(anchor.numel()),
                     'ess_fraction': float(ess_fraction.mean()),
                     'off_soft_error': float(off_error.mean()), 'on_soft_error': float(on_error.mean())}
        return (weights * pair_loss).sum() / valid.sum(), stats
