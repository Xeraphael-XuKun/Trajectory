"""Frozen identity coordinates; FP32, identity-summed, image-mean KL."""
import torch
import torch.nn.functional as F
from model.m2_teacher import normalized_clip_logits


@torch.no_grad()
def fixed_teacher_targets(logits, mods, views, shift, temperature=2.):
    with torch.cuda.amp.autocast(enabled=False):
        neutral = logits.float() + (.5-views.float())[:, None]*shift.float()[mods]
        return F.log_softmax(logits.float()/temperature, dim=-1), F.log_softmax(neutral/temperature, dim=-1)


def kl_rows(log_target, log_student):
    with torch.cuda.amp.autocast(enabled=False):
        target = log_target.detach().float()
        return (target.exp()*(target-log_student.float())).sum(-1)


def neutral_kd(g_student, projection, text, log_q, temperature=2.):
    with torch.cuda.amp.autocast(enabled=False):
        logits = normalized_clip_logits(g_student, projection, text, .07)
        return temperature**2*kl_rows(log_q, F.log_softmax(logits/temperature, dim=-1)).mean()


@torch.no_grad()
def target_quality(logp, logq, targets, mods, views, paired, log_student=None):
    """Training identities only, soft targets at KD temperature; per image + groups."""
    index = torch.arange(len(targets), device=targets.device)
    rows = {}
    distributions = [('raw', logp), ('neutral', logq)]
    if log_student is not None: distributions.append(('student', log_student.detach()))
    for name, log in distributions:
        true = log[index, targets]
        other = log.clone(); other[index, targets] = -torch.inf
        rows.update({name+'_true_probability':true.exp(), name+'_nll':-true,
            name+'_top1':(log.argmax(-1)==targets).float(),
            name+'_logit_margin':2*(true-other.max(-1).values),
            name+'_entropy':-(log.exp()*log).sum(-1)})
    mix = torch.logaddexp(logp, logq)-torch.log(torch.tensor(2., device=logp.device))
    rows['JS'] = .5*(kl_rows(logp,mix)+kl_rows(logq,mix))
    rows['L1'] = (logp.exp()-logq.exp()).abs().sum(-1)
    rows['delta_true_probability'] = rows['neutral_true_probability']-rows['raw_true_probability']
    output = dict(temperature=2., margin_units='base logits before distill temperature', per_image={k:v.cpu().tolist() for k,v in rows.items()}, groups=[])
    for m in range(3):
        for v in range(2):
            for coverage in (False, True):
                mask = (mods==m)&(views==v)&(paired==coverage)
                if mask.any():
                    output['groups'].append(dict(modality=m, platform=v, paired_coverage=coverage,
                        count=int(mask.sum()), **{k:float(x[mask].mean()) for k,x in rows.items()}))
    return output
