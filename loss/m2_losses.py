"""Losses specific to AS-ReID M2-2."""
import torch
import torch.nn.functional as F


def ground_loss(text_features, pids, rgb_centers, temperature=0.07):
    if rgb_centers is None or rgb_centers.numel() == 0:
        raise RuntimeError('M2-2 ground loss requires a non-empty RGB center cache')
    logits = text_features.float() @ rgb_centers.float().t() / float(temperature)
    return F.cross_entropy(logits, pids.long())


def xit_loss(visual_features, text_features, pids, modalities, temperature=0.07):
    """Multi-positive cross-spectrum NCE; text targets are already detached."""
    z, t = F.normalize(visual_features.float(), dim=-1), F.normalize(text_features.float(), dim=-1)
    pids, modalities = pids.long(), modalities.long()
    vals = []
    for i in range(z.shape[0]):
        cand = modalities != modalities[i]
        pos = cand & (pids == pids[i])
        neg = cand & (pids != pids[i])
        if pos.any() and neg.any():
            logits = (z[i] @ t[cand].t()) / float(temperature)
            positive = pos[cand]
            vals.append(-(torch.log_softmax(logits, dim=0)[positive].mean()))
    if not vals:
        return z.sum() * 0.0
    return torch.stack(vals).mean()

