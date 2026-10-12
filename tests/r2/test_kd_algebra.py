"""OneSided value/gradient/gate checks against independent full KL."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from loss.r2_kd import one_sided_kd
from model.m2_teacher import kd_kl


def reference(t,s,y,td=2.):
    keep=torch.ones_like(s,dtype=torch.bool)
    keep.scatter_(1,y[:,None],False)
    sn,tn=(s/td)[keep].reshape(len(y),-1),(t.detach()/td)[keep].reshape(len(y),-1)
    bs=F.log_softmax(torch.stack([(s/td).gather(1,y[:,None]).squeeze(1),sn.logsumexp(-1)],-1),-1)
    bt=F.log_softmax(torch.stack([(t.detach()/td).gather(1,y[:,None]).squeeze(1),tn.logsumexp(-1)],-1),-1)
    ns,nt=F.log_softmax(sn,-1),F.log_softmax(tn,-1)
    binary=(bt.exp()*(bt-bs)).sum(-1)
    weighted_tail=bt[:,1].exp()*(nt.exp()*(nt-ns)).sum(-1)
    logt,logs=F.log_softmax(t.detach()/td,-1),F.log_softmax(s/td,-1)
    full=td**2*(logt.exp()*(logt-logs)).sum(-1).mean()
    return binary,weighted_tail,bs,bt,full


def check(n,dtype,scale):
    torch.manual_seed(1234)
    t=(torch.randn(n,500,dtype=dtype)*scale).requires_grad_()
    s=(torch.randn(n,500,dtype=dtype)*scale).requires_grad_()
    y=torch.arange(n)%500
    mod=torch.arange(n)%3
    plat=torch.arange(n)%2
    atol,rtol=(1e-10,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
    b,tail,bs,bt,full=reference(t,s,y)
    parts=4*(b+tail).mean()
    torch.testing.assert_close(full,parts,atol=atol,rtol=rtol)
    torch.testing.assert_close(torch.autograd.grad(full,s,retain_graph=True)[0],
      torch.autograd.grad(parts,s,retain_graph=True)[0],atol=atol,rtol=rtol)
    if dtype==torch.float32:
        torch.testing.assert_close(full,kd_kl(t,s,2.),atol=atol,rtol=rtol)
    active=(bs[:,0].detach()<bt[:,0]).to(dtype)
    expected=4*(active*b+tail).mean()
    loss,stats=one_sided_kd(t,s,y,2.,True,modality=mod,platform=plat)
    torch.testing.assert_close(loss,expected,atol=atol,rtol=rtol)
    torch.testing.assert_close(torch.autograd.grad(loss,s,retain_graph=True)[0],
      torch.autograd.grad(expected,s,retain_graph=True)[0],atol=atol,rtol=rtol)
    loss.backward()
    assert t.grad is None and torch.isfinite(loss) and torch.isfinite(s.grad).all()
    assert sum(v['count'] for v in stats['teacher_correctness_x_active'].values())==n
    assert sum(v['count'] for v in stats['active_by_modality'].values())==n
    assert sum(v['active_count'] for v in stats['active_by_platform'].values())==int(active.sum())
    assert abs(stats['binary_raw']-stats['retained_binary_raw']-stats['canceled_binary_raw'])<atol+rtol*abs(stats['binary_raw'])
    print('PASS',n,str(dtype),scale,'KD',loss.item(),'decomposition_error',stats['decomposition_error'])


def gates(dtype):
    torch.manual_seed(93)
    t=torch.randn(7,500,dtype=dtype).requires_grad_()
    y=torch.arange(7)
    student=t.detach().clone()+torch.randn_like(t)*0.3
    keep=torch.ones_like(t,dtype=torch.bool)
    keep.scatter_(1,y[:,None],False)
    adjustment=2*((student/2)[keep].reshape(7,499).logsumexp(-1)-
                  (t.detach()/2)[keep].reshape(7,499).logsumexp(-1))
    atol,rtol=(1e-10,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
    for active in (True,False):
        s=student.clone()
        s.scatter_(1,y[:,None],(t.detach().gather(1,y[:,None]).squeeze(1)+adjustment+(-2. if active else 2.))[:,None])
        s.requires_grad_()
        loss,stats=one_sided_kd(t,s,y,2.,True)
        b,tail,bs,bt,full=reference(t,s,y)
        assert stats['active_fraction']==float(active)
        grad=torch.autograd.grad(loss,s,retain_graph=True)[0]
        if active:
            torch.testing.assert_close(loss,full,atol=atol,rtol=rtol)
            torch.testing.assert_close(grad,torch.autograd.grad(full,s)[0],atol=atol,rtol=rtol)
        else:
            torch.testing.assert_close(loss,4*tail.mean(),atol=atol,rtol=rtol)
            assert torch.count_nonzero(grad.gather(1,y[:,None]))==0
            assert grad[keep].abs().sum()>0
            assert (bt[:,1].exp()-(1-bs[:,0].exp())).abs().max()>1e-3
    # Exact boundary is inactive and does not divide by an empty active set.
    equal=t.detach().clone().requires_grad_()
    boundary,stats=one_sided_kd(t,equal,y,2.,True)
    assert stats['active_fraction']==0 and boundary.item()==0
    torch.testing.assert_close(torch.autograd.grad(boundary,equal)[0],torch.zeros_like(equal),atol=atol,rtol=rtol)
    # Nearby one-sided limits meet at the original tail objective.
    for delta in (-1e-5,1e-5):
        near=equal.detach().clone()
        near[torch.arange(7),y]+=delta
        value,_=one_sided_kd(t,near,y,2.)
        # At nearly zero KL, FP32 cancellation dominates the O(delta**2)
        # signal. Use the already declared dtype tolerance, without clamping
        # the production loss; FP64 remains the mathematical reference.
        assert torch.isfinite(value) and abs(value.item()-boundary.item())<atol
    print('OneSided active/inactive/boundary/continuity PASS',str(dtype))


if __name__=='__main__':
    for dtype in (torch.float64,torch.float32):
        for n in (1,7):
            check(n,dtype,3.)
        check(7,dtype,10000.)
        gates(dtype)
