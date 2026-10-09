"""M2-4: label-based post-BN centers and masked relation KL (§6)."""
import torch
import torch.nn.functional as F


def identity_modality_centers(features, pids, modalities, n_modalities=3):
    f = F.normalize(features.float(), dim=-1)
    ids, inverse = torch.unique(pids, sorted=True, return_inverse=True)
    index = inverse * n_modalities + modalities.long()
    centers = f.new_zeros(len(ids) * n_modalities, f.shape[-1]).index_add(0, index, f)
    counts = f.new_zeros(len(ids) * n_modalities).index_add(0, index, f.new_ones(len(f)))
    centers = F.normalize(centers / counts.clamp_min(1)[:, None], dim=-1)
    return centers.reshape(len(ids), n_modalities, -1), ids, counts.reshape(len(ids), n_modalities) > 0


def relation_kl(centers, teacher_cos, tau_student=.07, tau_teacher=.07, valid=None, return_stats=False):
    p, modalities, _ = centers.shape
    if valid is None:
        valid = torch.ones(p, modalities, dtype=torch.bool, device=centers.device)
    terms, stats = [], {}
    for m in range(modalities):
        for n in range(modalities):
            losses, ht, hs = [], [], []
            similarities = centers[:, m].float() @ centers[:, n].float().t()
            for y in range(p):
                candidates = valid[:, n].clone()
                candidates[y] = False
                if p < 3 or not valid[y, m] or candidates.sum() < 2:
                    continue
                lt = F.log_softmax(teacher_cos[y, candidates].float().detach() / tau_teacher, dim=0)
                ls = F.log_softmax(similarities[y, candidates] / tau_student, dim=0)
                losses.append((lt.exp() * (lt - ls)).sum())
                ht.append(-(lt.exp() * lt).sum()); hs.append(-(ls.exp() * ls).sum())
            if losses:
                terms.extend(losses)
                stats['kl_{}_{}'.format(m,n)] = torch.stack(losses).mean().detach().item()
                stats['teacher_entropy_{}_{}'.format(m,n)] = torch.stack(ht).mean().item()
                stats['student_entropy_{}_{}'.format(m,n)] = torch.stack(hs).mean().detach().item()
    loss = torch.stack(terms).mean() if terms else centers.sum() * 0
    stats['center_norm'] = centers.norm(dim=-1).mean().detach().item()
    return (loss, stats) if return_stats else loss
