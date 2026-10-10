"""M2-4 post-BN centers and nine ordered, row-normalized KL relations."""
import torch
import torch.nn.functional as F

def identity_modality_centers(features, pids, modalities, n_modalities=3):
    with torch.cuda.amp.autocast(enabled=False):
        f = F.normalize(features.float(), dim=-1)
        ids, inverse = torch.unique(pids, sorted=True, return_inverse=True)
        index = inverse * n_modalities + modalities.long()
        centers = f.new_zeros(len(ids) * n_modalities, f.shape[-1]).index_add(0, index, f)
        counts = f.new_zeros(len(ids) * n_modalities).index_add(0, index, f.new_ones(len(f)))
        centers = F.normalize(centers / counts.clamp_min(1)[:, None], dim=-1)
        return centers.reshape(len(ids), n_modalities, -1), ids, counts.reshape(len(ids), n_modalities) > 0

def relation_kl(centers, teacher_cos, tau_student=.07, tau_teacher=.07, valid=None, return_stats=False):
    with torch.cuda.amp.autocast(enabled=False):
        centers = centers.float()
        teacher = teacher_cos.detach().float()
        p, modalities, _ = centers.shape
        if valid is None:
            valid = torch.ones(p, modalities, dtype=torch.bool, device=centers.device)
        loss_sum, count = centers.sum() * 0., 0
        stats = {key: centers.new_full((modalities, modalities), float('nan'))
                 for key in ('kl', 'teacher_entropy', 'student_entropy')}
        stats['rows'] = torch.zeros(modalities, modalities, dtype=torch.long, device=centers.device)
        if p >= 3:
            off_diagonal = ~torch.eye(p, dtype=torch.bool, device=centers.device)
            for m in range(modalities):
                for n in range(modalities):
                    candidates = off_diagonal & valid[:, n][None]
                    rows = valid[:, m] & candidates.any(dim=1)
                    if not rows.any():
                        continue
                    allowed = candidates[rows]
                    # Only nonempty rows enter softmax. Mask log values before subtraction;
                    # excluded entries must never evaluate 0*(-inf - -inf).
                    tlog_raw = F.log_softmax((teacher[rows] / tau_teacher).masked_fill(~allowed, float('-inf')), dim=1)
                    similarity = centers[rows, m] @ centers[:, n].t()
                    slog_raw = F.log_softmax((similarity / tau_student).masked_fill(~allowed, float('-inf')), dim=1)
                    tlog, slog = tlog_raw.masked_fill(~allowed, 0.), slog_raw.masked_fill(~allowed, 0.)
                    tp, sp = tlog_raw.exp(), slog_raw.exp()
                    kl = (tp * (tlog - slog)).sum(1)
                    loss_sum = loss_sum + kl.sum()
                    count += len(kl)
                    if return_stats:
                        stats['kl'][m, n] = kl.detach().mean()
                        stats['teacher_entropy'][m, n] = -(tp * tlog).sum(1).mean()
                        stats['student_entropy'][m, n] = -(sp.detach() * slog.detach()).sum(1).mean()
                        stats['rows'][m, n] = len(kl)
        loss = loss_sum / count if count else loss_sum
        if return_stats:
            stats['center_norm'] = centers[valid].norm(dim=-1).mean().detach() if valid.any() else centers.new_zeros(())
            return loss, stats
        return loss
