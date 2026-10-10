"""M2-1 fixed condition prototype sets; all identities, true missing masks."""
import torch
import torch.nn.functional as F
from utils.m1_artifacts import validate_bank

def _layout(pids, ids):
    groups = [torch.nonzero(pids == y, as_tuple=False).flatten() for y in ids]
    width = max(len(group) for group in groups)
    index = torch.zeros(len(ids), width, dtype=torch.long, device=pids.device)
    valid = torch.zeros_like(index, dtype=torch.bool)
    for row, group in enumerate(groups):
        index[row, :len(group)] = group
        valid[row, :len(group)] = True
    return index, valid

def _group(scores, layout):
    index, valid = layout
    values = scores[:, index].masked_fill(~valid[None], float('-inf'))
    return values.logsumexp(dim=-1) - valid.sum(1).float().log()

def prepare_bank(bank):
    # Fixed bank grouping is prepared once, without N*Y Python CE calls.
    pids, mods = bank['pid'], bank['modality']
    ids = pids.unique(sorted=True)
    bank['_ids'], bank['_all'] = ids, _layout(pids, ids)
    bank['_cross'] = {}
    for m in range(3):
        keep = mods != m
        candidates = pids[keep].unique(sorted=True)
        if candidates.numel() >= 2:
            bank['_cross'][m] = (keep, candidates, _layout(pids[keep], candidates))
    return bank

def load_text_bank(path, cfg, num_classes, device='cpu'):
    bank = torch.load(path, map_location='cpu', weights_only=False)
    validate_bank(bank, cfg, num_classes)
    bank['text'] = bank['text'].detach().to(device)
    for key in ('pid', 'modality', 'platform'):
        bank[key] = bank[key].long().to(device)
    return prepare_bank(bank)

def condition_bridge_loss(z, labels, modalities, bank, temperature=.07,
                          cross_temperature=.07, all_weight=.5, cross_weight=.5):
    with torch.cuda.amp.autocast(enabled=False):
        if '_all' not in bank:
            prepare_bank(bank)
        z = F.normalize(z.float(), dim=1)
        text = F.normalize(bank['text'].detach().float(), dim=1)
        pids, ids = bank['pid'], bank['_ids']
        if not torch.isin(labels, ids).all():
            raise ValueError('M2-1 bank is missing a current train identity')
        similarity = z @ text.t()
        target = torch.searchsorted(ids, labels)
        all_logits = _group(similarity / temperature, bank['_all'])
        la = F.cross_entropy(all_logits, target)
        cross_sum, valid_count = z.sum() * 0., 0
        accuracies, contributions = [], []
        for m in range(3):
            rows = modalities == m
            if not rows.any():
                accuracies.append(z.new_tensor(float('nan')))
                contributions.append(z.new_full((3, 2), float('nan')))
                continue
            accuracies.append((ids[all_logits[rows].argmax(1)] == labels[rows]).float().mean())
            # Distribution of softmax contributions within the true ID set.
            truth = pids[None] == labels[rows, None]
            weights = torch.softmax((similarity[rows] / temperature).masked_fill(~truth, float('-inf')), dim=1)
            distribution = torch.zeros(3, 2, device=z.device)
            for mod in range(3):
                for view in range(2):
                    distribution[mod, view] = weights[:, (bank['modality'] == mod) & (bank['platform'] == view)].sum(1).mean()
            contributions.append(distribution)
            if m not in bank['_cross']:
                continue
            keep, candidates, layout = bank['_cross'][m]
            selected = rows & torch.isin(labels, candidates)
            if not selected.any():
                continue
            logits = _group(similarity[selected][:, keep] / cross_temperature, layout)
            targets = torch.searchsorted(candidates, labels[selected])
            cross_sum = cross_sum + F.cross_entropy(logits, targets, reduction='sum')
            valid_count += int(selected.sum())
        lx = cross_sum / valid_count if valid_count else cross_sum
        return all_weight * la + cross_weight * lx, {
            'all': la.detach(), 'cross': lx.detach(), 'valid': valid_count,
            'accuracy_by_modality': torch.stack(accuracies).detach(),
            'prototype_contributions': torch.stack(contributions).detach()}
