"""I3: branch mean, identity sum, image mean; temperature compensation once."""
import torch
from model.m2_teacher import normalized_clip_logits
from loss.iclr_neutral import kl_rows


def paired_terms(g_on, g_off, projection, text, log_p, log_q, temperature=2.):
    with torch.cuda.amp.autocast(enabled=False):
        log_on = (normalized_clip_logits(g_on, projection, text)/temperature).log_softmax(-1)
        log_off = (normalized_clip_logits(g_off, projection, text)/temperature).log_softmax(-1)
        raw_on = kl_rows(log_q, log_on).mean()
        raw_off = kl_rows(log_p, log_off).mean()
        return temperature**2 * .5 * (raw_on + raw_off), raw_off, raw_on


@torch.no_grad()
def branch_quality(log_on, log_off, log_p, log_q, mods, views, shift):
    # Logs are at Td=2; return centered delta in base-logit units.
    delta_s = 2*(log_on-log_off)
    delta_s -= delta_s.mean(-1, keepdim=True)
    delta_t = (.5-views.float())[:,None]*shift[mods].float()
    s_norm,t_norm = delta_s.norm(dim=-1),delta_t.norm(dim=-1)
    valid = (s_norm>0)&(t_norm>0)
    cosine = torch.zeros_like(s_norm); ratio=torch.zeros_like(s_norm)
    cosine[valid]=(delta_s[valid]*delta_t[valid]).sum(-1)/(s_norm[valid]*t_norm[valid])
    ratio[t_norm>0]=s_norm[t_norm>0]/t_norm[t_norm>0]
    def js(a,b):
        mixture=torch.logaddexp(a,b)-a.new_tensor(2.).log()
        return .5*(kl_rows(a,mixture)+kl_rows(b,mixture))
    values=dict(off_to_raw=kl_rows(log_p,log_off),on_to_neutral=kl_rows(log_q,log_on),
        off_to_neutral=kl_rows(log_q,log_off),on_to_raw=kl_rows(log_p,log_on),
        student_JS=js(log_on,log_off),teacher_JS=js(log_p,log_q),
        student_delta_norm=s_norm,target_delta_norm=t_norm)
    rows={k:v.cpu().tolist() for k,v in values.items()}
    rows['direction_cosine']=[float(cosine[i]) if valid[i] else None for i in range(len(views))]
    rows['delta_norm_ratio']=[float(ratio[i]) if t_norm[i]>0 else None for i in range(len(views))]
    groups=[]
    for m in range(3):
        for v in range(2):
            mask=(mods==m)&(views==v);direction=mask&valid;nonzero=mask&(t_norm>0)
            if not mask.any():continue
            groups.append(dict(modality=m,platform=v,count=int(mask.sum()),
                **{k:float(x[mask].mean()) for k,x in values.items()},
                direction_valid=int(direction.sum()),
                direction_cosine=float(cosine[direction].mean()) if direction.any() else None,
                delta_norm_ratio=float(ratio[nonzero].mean()) if nonzero.any() else None))
    return dict(temperature=2.,delta_units='centered base logits before Td',per_image=rows,groups=groups)
