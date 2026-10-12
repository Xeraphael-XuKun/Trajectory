"""E2 only: one-sided true-ID binary KL plus the unchanged teacher tail."""
import torch
import torch.nn.functional as F


def one_sided_kd(logits_teacher, logits_student, target, temperature=2.,
                 log_statistics=False, modality=None, platform=None):
    """Batch mean, Td**2 once; outer lambda=0.5 remains in processor.

    Inputs are the original cosine/0.07 logits. Only teacher and gate are
    detached. Tail weight is 1-aT, including when the binary gate is closed.
    Training probability math is FP32; FP64 is retained for reference tests.
    """
    with torch.cuda.amp.autocast(enabled=False):
        dtype=torch.float64 if logits_student.dtype==torch.float64 else torch.float32
        td=float(temperature)
        t=logits_teacher.detach().to(dtype)/td
        s=logits_student.to(dtype)/td
        n,k=s.shape
        keep=torch.ones_like(s,dtype=torch.bool)
        keep.scatter_(1,target[:,None],False)
        sn,tn=s[keep].reshape(n,k-1),t[keep].reshape(n,k-1)
        sy,ty=s.gather(1,target[:,None]).squeeze(1),t.gather(1,target[:,None]).squeeze(1)
        bs=F.log_softmax(torch.stack([sy,sn.logsumexp(-1)],-1),-1)
        bt=F.log_softmax(torch.stack([ty,tn.logsumexp(-1)],-1),-1)
        ns,nt=F.log_softmax(sn,-1),F.log_softmax(tn,-1)
        binary=(bt.exp()*(bt-bs)).sum(-1)
        tail=(nt.exp()*(nt-ns)).sum(-1)
        mass=bt[:,1].exp()
        active=(bs[:,0].detach()<bt[:,0]).to(dtype)
        per_image=active*binary+mass*tail
        loss=td*td*per_image.mean()
        stats=None
        if log_statistics:
            with torch.no_grad():
                lt,ls=F.log_softmax(t,-1),F.log_softmax(s.detach(),-1)
                original_full=(lt.exp()*(lt-ls)).sum(-1)
                correct=t.argmax(-1)==target
                def groups(labels,count):
                    result={}
                    for index in range(count):
                        selected=labels==index
                        size=int(selected.sum().item())
                        result[str(index)]=dict(count=size,active_count=int(active[selected].sum().item()),
                            active_fraction=active[selected].mean().item() if size else None)
                    return result
                joint={}
                for is_correct,label in ((True,'correct'),(False,'wrong')):
                    for is_active,gate in ((True,'active'),(False,'inactive')):
                        size=int(((correct==is_correct)&(active.bool()==is_active)).sum().item())
                        joint[label+'_'+gate]=dict(count=size,batch_fraction=size/n)
                stats=dict(batch_size=n,active_fraction=active.mean().item(),
                    teacher_p_y=bt[:,0].exp().mean().item(),student_p_y=bs[:,0].exp().mean().item(),
                    teacher_entropy=-(lt.exp()*lt).sum(-1).mean().item(),
                    binary_raw=binary.mean().item(),retained_binary_raw=(active*binary).mean().item(),
                    canceled_binary_raw=((1-active)*binary).mean().item(),
                    teacher_tail_raw=tail.mean().item(),weighted_teacher_tail_raw=(mass*tail).mean().item(),
                    original_full_kd_Td2=(td*td*original_full.mean()).item(),kd_Td2=loss.detach().item(),
                    decomposition_error=(binary+mass*tail-original_full).abs().max().item(),
                    minimum_per_image_raw=per_image.detach().min().item(),
                    teacher_correctness_x_active=joint)
                if modality is not None:
                    stats['active_by_modality']=groups(modality,3)
                if platform is not None:
                    stats['active_by_platform']=groups(platform,2)
        return loss,stats
