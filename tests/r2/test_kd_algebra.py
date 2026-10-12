"""Required E1 probability/value/gradient checks; no model or artifact loads."""
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from loss.r2_kd import flat_tail_log_target, flat_tail_kd
from model.m2_teacher import kd_kl


def check(n,dtype,scale):
    torch.manual_seed(1234)
    t=(torch.randn(n,500,dtype=dtype)*scale).requires_grad_()
    s=(torch.randn(n,500,dtype=dtype)*scale).requires_grad_()
    y=torch.arange(n,dtype=torch.long)%500
    td=2.
    logq,bt=flat_tail_log_target(t,y,td)
    q=logq.exp()
    assert not q.requires_grad
    atol,rtol=(1e-10,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
    torch.testing.assert_close(q.sum(-1),torch.ones(n,dtype=dtype),atol=atol,rtol=rtol)
    py=F.softmax(t.detach()/td,-1).gather(1,y[:,None]).squeeze(1)
    torch.testing.assert_close(q.gather(1,y[:,None]).squeeze(1),py,atol=atol,rtol=rtol)
    keep=torch.ones_like(s,dtype=torch.bool)
    keep.scatter_(1,y[:,None],False)
    tailq=q[keep].reshape(n,499)
    torch.testing.assert_close(tailq,tailq[:,:1].expand_as(tailq),atol=0,rtol=0)
    loss,stats=flat_tail_kd(t,s,y,td,True)
    ls=F.log_softmax(s/td,-1)
    direct=td**2*(q*(logq-ls)).sum(-1).mean()
    torch.testing.assert_close(loss,direct,atol=atol,rtol=rtol)
    sn=(s/td)[keep].reshape(n,499)
    tn=(t.detach()/td)[keep].reshape(n,499)
    bs=F.log_softmax(torch.stack([(s/td).gather(1,y[:,None]).squeeze(1),sn.logsumexp(-1)],-1),-1)
    ns,nt=F.log_softmax(sn,-1),F.log_softmax(tn,-1)
    binary=(bt.exp()*(bt-bs)).sum(-1)
    mass=bt[:,1].exp()
    decomposed=td**2*(binary+mass*(-math.log(499)-ns.mean(-1))).mean()
    torch.testing.assert_close(loss,decomposed,atol=atol,rtol=rtol)
    g=torch.autograd.grad(loss,s,retain_graph=True)[0]
    dg=torch.autograd.grad(decomposed,s,retain_graph=True)[0]
    torch.testing.assert_close(g,dg,atol=atol,rtol=rtol)
    original=td**2*(F.log_softmax(t.detach()/td,-1).exp()*(F.log_softmax(t.detach()/td,-1)-ls)).sum(-1).mean()
    original_parts=td**2*(binary+mass*(nt.exp()*(nt-ns)).sum(-1)).mean()
    torch.testing.assert_close(original,original_parts,atol=atol,rtol=rtol)
    torch.testing.assert_close(torch.autograd.grad(original,s,retain_graph=True)[0],
                               torch.autograd.grad(original_parts,s,retain_graph=True)[0],atol=atol,rtol=rtol)
    if dtype==torch.float32:
        torch.testing.assert_close(original,kd_kl(t,s,td),atol=atol,rtol=rtol)
    loss.backward()
    assert t.grad is None and torch.isfinite(s.grad).all() and torch.isfinite(loss)
    assert math.isfinite(stats['teacher_entropy']) and math.isfinite(stats['flat_tail_entropy'])
    print('PASS',n,str(dtype),scale,'KD',loss.item(),'decomposition_error',stats['decomposition_error'])


if __name__=='__main__':
    for dtype in (torch.float64,torch.float32):
        for n in (1,7):
            check(n,dtype,3.)
        check(7,dtype,10000.)
    # Wrong teacher peak may become a true-class peak after flattening.
    t=torch.zeros(1,500)
    t[0,1]=8.
    t[0,0]=6.
    q,_=flat_tail_log_target(t,torch.tensor([0]),2.)
    assert t.argmax(-1).item()==1 and q.argmax(-1).item()==0
    print('FlatTail rewritten argmax case: PASS')
