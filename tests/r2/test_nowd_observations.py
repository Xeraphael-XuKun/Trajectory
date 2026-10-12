"""Check that observations preserve controller forward/gradients/RNG."""
import sys,copy
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.backbones.token_trajectory import DeepTextC0Controller
from utils.r2_experiment import parameter_snapshot,gradient_snapshot


def main():
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    control=DeepTextC0Controller(12,129)
    with torch.no_grad():
        control.context.normal_()
        control.o.weight.normal_(std=.01)
    observed=copy.deepcopy(control)
    observed.r2_collect_stats=True
    velocity=torch.randn(3,129,768,requires_grad=True)
    v2=velocity.detach().clone().requires_grad_()
    correction=torch.randn(3,129,768,requires_grad=True)*.01
    correction=correction.detach().requires_grad_()
    c2=correction.detach().clone().requires_grad_()
    rng=torch.get_rng_state().clone()
    result=control(1,velocity,correction)
    result.square().mean().backward()
    seen=observed(1,v2,c2)
    seen.square().mean().backward()
    assert torch.equal(result,seen) and torch.equal(rng,torch.get_rng_state())
    assert torch.equal(velocity.grad,v2.grad) and torch.equal(correction.grad,c2.grad)
    for a,b in zip(control.parameters(),observed.parameters()):
        assert torch.equal(a.grad,b.grad)
    with torch.no_grad():
        k,v=observed.key_values()
        att=F.softmax(observed.q(F.layer_norm(v2,(768,)))@k[0].t()/8.,-1)
        multiplier=1+observed.rho*torch.tanh(observed.o(att@v[0]))
        stat=observed.r2_last_stats[0]
        torch.testing.assert_close(stat[4],-(att*att.log()).sum(-1).mean())
        torch.testing.assert_close(stat[5],multiplier.var(0,unbiased=False).mean())
        torch.testing.assert_close(stat[6],(seen-c2).norm()/(c2.norm()+1e-12))
        assert stat[5]>0 and stat[6]>0 and .75<=stat[2]<=stat[3]<=1.25
    owner=SimpleNamespace(base=SimpleNamespace(deep_c0_controller=observed))
    with torch.no_grad():
        observed.q.weight.fill_(1e-41)
    assert parameter_snapshot(owner)['q.weight']['norm_fp64']>0
    observed.q.weight.grad=None
    observed.k.weight.grad=torch.zeros_like(observed.k.weight)
    observed.v.weight.grad=torch.full_like(observed.v.weight,1e-41)
    snapshot=gradient_snapshot(owner)
    assert snapshot['q.weight']['status']=='None'
    assert snapshot['k.weight']['status']=='all_zero'
    assert snapshot['v.weight']['status']=='finite_nonzero' and snapshot['v.weight']['norm_fp64']>0
    print('CONTROLLER_OBSERVATIONS_FORWARD_GRAD_RNG_AND_FP64_PASS')


if __name__=='__main__':
    main()
