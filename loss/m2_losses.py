"""Losses specific to AS-ReID M2-2."""
import torch
import torch.nn.functional as F


def ground_loss(text_features, pids, rgb_centers, temperature=0.07):
    if rgb_centers is None or rgb_centers.numel() == 0:
        raise RuntimeError('M2-2 ground loss requires a non-empty RGB center cache')
    with torch.cuda.amp.autocast(enabled=False):
        logits = text_features.float() @ rgb_centers.float().t() / float(temperature)
        return F.cross_entropy(logits, pids.long())


def xit_loss(visual_features, text_features, pids, modalities, temperature=0.07):
    """Multi-positive cross-spectrum NCE; text targets are already detached."""
    with torch.cuda.amp.autocast(enabled=False):
        z = F.normalize(visual_features.float(), dim=-1)
        t = F.normalize(text_features.detach().float(), dim=-1)
        pids, modalities = pids.long(), modalities.long()
        candidates = modalities[:, None] != modalities[None, :]
        positive = candidates & (pids[:, None] == pids[None, :])
        negative = candidates & (pids[:, None] != pids[None, :])
        valid = positive.any(1) & negative.any(1)
        if not valid.any():
            return z.sum() * 0.0
        logits = (z[valid] @ t.t()) / float(temperature)
        logits = logits.masked_fill(~candidates[valid], -torch.inf)
        log_probability = torch.log_softmax(logits, dim=1)
        terms = log_probability.masked_fill(~positive[valid], 0.0)
        return -(terms.sum(1) / positive[valid].sum(1)).mean()
