import torch
import torch.nn.functional as F


def load_text_bank(path, device=None):
    bank = torch.load(path, map_location=device or 'cpu', weights_only=False)
    if not isinstance(bank, dict) or not {'text', 'pid', 'modality'}.issubset(bank):
        raise ValueError('M2-1 text bank must contain text, pid and modality')
    text = bank['text']
    pid = torch.as_tensor(bank['pid'])
    modality = torch.as_tensor(bank['modality'])
    if text.ndim != 2 or text.shape[1] != 512 or pid.ndim != 1 or modality.ndim != 1:
        raise ValueError('M2-1 text bank has incompatible shapes')
    if text.shape[0] != pid.numel() or pid.numel() != modality.numel():
        raise ValueError('M2-1 text bank fields have different lengths')
    if pid.numel() == 0 or (pid < 0).any() or (modality < 0).any() or (modality > 2).any():
        raise ValueError('M2-1 text bank contains invalid train mappings')
    return bank


def _group(scores, pids, ids):
    """Log-mean-exp over the bank entries for each identity."""
    return torch.stack([
        torch.logsumexp(scores[:, pids == y], dim=1)
        - torch.log((pids == y).sum().to(scores).float())
        for y in ids
    ], dim=1)


def condition_bridge_loss(z, labels, modalities, bank, temperature=.07,
                          all_weight=.5, cross_weight=.5):
    text = F.normalize(bank['text'].float().to(z.device), dim=1)
    pids = torch.as_tensor(bank['pid'], device=z.device).long()
    mods = torch.as_tensor(bank['modality'], device=z.device).long()
    labels = torch.as_tensor(labels, device=z.device).long()
    modalities = torch.as_tensor(modalities, device=z.device).long()
    z = F.normalize(z.float(), dim=1)
    ids = pids.unique(sorted=True)
    scores = z @ text.t() / temperature

    valid = torch.isin(labels, ids)
    if valid.any():
        target = torch.searchsorted(ids, labels[valid])
        la = F.cross_entropy(_group(scores[valid], pids, ids), target)
    else:
        la = z.sum() * 0.0

    vals = []
    for i, m in enumerate(modalities.tolist()):
        keep = mods != int(m)
        cand = pids[keep].unique(sorted=True)
        if cand.numel() < 2 or not bool(torch.isin(labels[i], cand)):
            continue
        q = scores[i, keep]
        pp = pids[keep]
        g = torch.stack([
            torch.logsumexp(q[pp == y], dim=0)
            - torch.log((pp == y).sum().to(q).float())
            for y in cand
        ])
        vals.append(F.cross_entropy(g[None], torch.searchsorted(cand, labels[i]).reshape(1)))
    lx = torch.stack(vals).mean() if vals else z.sum() * 0.0
    return all_weight * la + cross_weight * lx, {
        'all': la.detach(), 'cross': lx.detach(), 'valid': len(vals)
    }