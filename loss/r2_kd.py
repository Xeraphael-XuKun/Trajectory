"""E1 FlatTail only: unchanged true-ID probability, uniform non-target tail."""
import math
import torch
import torch.nn.functional as F


def flat_tail_log_target(logits_teacher, target, temperature=2.):
    """Inputs already include the base 0.07 temperature; detach the whole q."""
    dtype = torch.float64 if logits_teacher.dtype == torch.float64 else torch.float32
    t = logits_teacher.detach().to(dtype) / float(temperature)
    n, k = t.shape
    keep = torch.ones_like(t, dtype=torch.bool)
    keep.scatter_(1, target[:, None], False)
    tn = t[keep].reshape(n, k-1)
    ty = t.gather(1, target[:, None]).squeeze(1)
    bt = F.log_softmax(torch.stack([ty, tn.logsumexp(-1)], -1), -1)
    logq = (bt[:, 1:2] - math.log(k-1)).expand_as(t).clone()
    logq.scatter_(1, target[:, None], bt[:, 0:1])
    return logq, bt


def flat_tail_kd(logits_teacher, logits_student, target, temperature=2., log_statistics=False):
    """Return batch mean containing Td**2 once; caller applies lambda=0.5 once.

    Training probability math is FP32 even under AMP. FP64 inputs are retained
    only for algebra/gradient reference checks. Original full KD is diagnostic.
    """
    with torch.cuda.amp.autocast(enabled=False):
        dtype = torch.float64 if logits_student.dtype == torch.float64 else torch.float32
        teacher = logits_teacher.detach().to(dtype)
        student = logits_student.to(dtype)
        td = float(temperature)
        logq, bt = flat_tail_log_target(teacher, target, td)
        ls = F.log_softmax(student/td, -1)
        per_image = (logq.exp()*(logq-ls)).sum(-1)
        loss = td*td*per_image.mean()
        stats = None
        if log_statistics:
            with torch.no_grad():
                n, k = student.shape
                s = student.detach()/td
                t = teacher/td
                keep = torch.ones_like(s, dtype=torch.bool)
                keep.scatter_(1, target[:, None], False)
                sn = s[keep].reshape(n,k-1)
                sy = s.gather(1,target[:,None]).squeeze(1)
                bs = F.log_softmax(torch.stack([sy,sn.logsumexp(-1)],-1),-1)
                ns = F.log_softmax(sn,-1)
                binary = (bt.exp()*(bt-bs)).sum(-1)
                mass = bt[:,1].exp()
                uniform_tail = -math.log(k-1)-ns.mean(-1)
                lt = F.log_softmax(t,-1)
                original_full = (lt.exp()*(lt-ls.detach())).sum(-1)
                stats = dict(
                    teacher_entropy=-(lt.exp()*lt).sum(-1).mean().item(),
                    flat_tail_entropy=-(logq.exp()*logq).sum(-1).mean().item(),
                    teacher_p_y=bt[:,0].exp().mean().item(),
                    student_p_y=bs[:,0].exp().mean().item(),
                    rewritten_argmax_fraction=((t.argmax(-1)!=target)&
                        (bt[:,0]>bt[:,1]-math.log(k-1))).to(dtype).mean().item(),
                    binary_raw=binary.mean().item(),
                    uniform_tail_raw=uniform_tail.mean().item(),
                    weighted_uniform_tail_raw=(mass*uniform_tail).mean().item(),
                    original_full_kd_Td2=(td*td*original_full.mean()).item(),
                    kd_Td2=loss.detach().item(),
                    decomposition_error=(binary+mass*uniform_tail-per_image.detach()).abs().max().item(),
                    minimum_per_image_raw=per_image.detach().min().item())
        return loss, stats
