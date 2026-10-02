"""Focused checks for original-method composition and adjacent-block routing."""
from pathlib import Path
import subprocess
import types

import pytest
import torch

from config import cfg
from model.backbones.vit_pytorch import TransReID
from model.backbones.token_trajectory_variants import TrajectoryVariant, validate_trajectory_config
from model.backbones.token_trajectory_combinations import GatedTrajectoryCombination
from model.make_model import build_transformer
from solver import make_optimizer

ROOT=Path(__file__).resolve().parents[1]
CHOICES={'A':('adaptive_token_gate',1.0), 'B':('split_token_gate',1.0),
         'C':('velocity_cross_depth',0.0), 'D':('velocity_late_depth',0.0)}


def backbone(variant='dense', mix=0.0, enabled=True, factory=TransReID):
    return factory(img_size=(32,16),patch_size=16,stride_size=16,embed_dim=8,
                   depth=12,num_heads=2,mlp_ratio=2,drop_path_rate=0.2,
                   token_trajectory=enabled,token_trajectory_variant=variant,
                   token_trajectory_accel_mix=mix)


@pytest.mark.parametrize('variant,original',[('adaptive_token_gate','adaptive'),('split_token_gate','split')])
def test_composition_matches_original_direction_times_original_t4_gate(variant,original):
    torch.manual_seed(7)
    combined=GatedTrajectoryCombination(4,3,8,variant)
    base=TrajectoryVariant(4,3,8,original)
    gate=TrajectoryVariant(4,3,8,'token_gate')
    with torch.no_grad():
        for name,p in base.named_parameters():
            p.normal_(std=0.1)
            dict(combined.named_parameters())[name].copy_(p)
    v,old=torch.randn(2,3,8,requires_grad=True),torch.randn(2,3,8,requires_grad=True)
    # Original T4 starts at multiplier=1; the nonzero T1/T2 computation is intact.
    for layer,history in ((1,None),(2,old)):
        assert torch.equal(combined(layer,v,history),base(layer,v,history))
    with torch.no_grad():
        for p in gate.conditioners.parameters():p.normal_(std=0.1)
    combined.conditioners.load_state_dict(gate.conditioners.state_dict())
    for layer,history in ((1,None),(2,old)):
        a_norm=torch.zeros_like(v) if history is None else gate.normalize(v-history)
        features=torch.cat((gate.normalize(v),a_norm),dim=-1)
        multiplier=0.5+gate.conditioners[layer-1](features).float().sigmoid()
        expected=base(layer,v,history)*multiplier
        actual=combined(layer,v,history)
        assert torch.equal(actual,expected)
        grad_a=torch.autograd.grad(actual.square().sum(),(v,old),retain_graph=True,allow_unused=True)
        grad_b=torch.autograd.grad(expected.square().sum(),(v,old),retain_graph=True,allow_unused=True)
        for a,b in zip(grad_a,grad_b):
            # Shared vs separately computed acceleration changes FP32 gradient
            # accumulation order; forward values above must remain bitwise equal.
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a,b,rtol=1e-6,atol=3e-8)
    combined(2,v,old).square().sum().backward()
    names=['gain','conditioners.1.2.weight']+(['beta_coeff'] if original=='adaptive' else ['accel_gain'])
    for name in names:
        grad=dict(combined.named_parameters())[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0,name
    assert torch.count_nonzero(combined(2,v,old,gate=torch.zeros(2)))==0


@pytest.mark.parametrize('group',['C','D'])
def test_sparse_uses_three_immediately_preceding_velocities_and_matches_masked_c0(group):
    variant,mix=CHOICES[group]
    torch.manual_seed(3);sparse=backbone(variant,mix).train()
    torch.manual_seed(3);dense=backbone().train()
    actuator=sparse.token_trajectory
    assert actuator.active_layers==((4,7,11) if group=='C' else (9,10,11))
    assert actuator.gain.shape==(3,1,3,8)
    with torch.no_grad():
        actuator.gain.normal_(std=.05)
        for row,layer in enumerate(actuator.active_layers):
            dense.token_trajectory.gain[layer-1].copy_(actuator.gain[row])
    inputs,outputs,seen,hooks=[],[],[],[]
    for block in sparse.blocks:
        hooks.append(block.register_forward_pre_hook(lambda m,args:inputs.append(args[0].detach().clone())))
        hooks.append(block.register_forward_hook(lambda m,args,out:outputs.append(out.detach().clone())))
    hooks.append(actuator.register_forward_pre_hook(lambda m,args:seen.append((args[0],args[1].detach().clone()))))
    x=torch.randn(6,3,32,16)
    torch.manual_seed(11);actual=sparse(x);rng=torch.random.get_rng_state()
    for h in hooks:h.remove()
    torch.manual_seed(11);expected=dense(x)
    assert torch.equal(actual,expected) and torch.equal(rng,torch.random.get_rng_state())
    assert tuple(layer for layer,v in seen)==actuator.active_layers
    for layer,v in seen:
        assert torch.equal(v,outputs[layer-1]-inputs[layer-1])
    actual.square().mean().backward();expected.square().mean().backward()
    for name,p in sparse.named_parameters():
        if 'token_trajectory' not in name:
            q=dict(dense.named_parameters())[name]
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad),name
    for row,layer in enumerate(actuator.active_layers):
        assert torch.equal(actuator.gain.grad[row],dense.token_trajectory.gain.grad[layer-1])


@pytest.mark.parametrize('group',CHOICES)
def test_configuration_initialization_learning_optimizer_and_checkpoint(group,tmp_path):
    c=cfg.clone();c.merge_from_file(str(ROOT/('configs/trajectory_'+group+'.yml')))
    validate_trajectory_config(c)
    variant,mix=CHOICES[group]
    assert c.MODEL.TOKEN_TRAJECTORY_VARIANT==variant and c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX==mix
    assert c.TEST.NECK_FEAT=='after'
    bad=c.clone();bad.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX=.5
    with pytest.raises(ValueError):validate_trajectory_config(bad)
    torch.manual_seed(19);reference=backbone(enabled=False);rng=torch.random.get_rng_state()
    torch.manual_seed(19);model=torch.nn.Module();model.base=backbone(variant,mix)
    assert torch.equal(rng,torch.random.get_rng_state())
    for key,value in reference.state_dict().items():assert torch.equal(value,model.base.state_dict()[key]),key
    x=torch.randn(6,3,32,16)
    torch.manual_seed(23);expected=reference(x);rng=torch.random.get_rng_state()
    torch.manual_seed(23);actual=model.base(x)
    assert torch.equal(actual,expected) and torch.equal(rng,torch.random.get_rng_state())
    optimizer,_=make_optimizer(c,model,torch.nn.Linear(8,2))
    for pg in optimizer.param_groups:assert pg['weight_decay']==1e-4
    for _ in range(3):
        optimizer.zero_grad();loss=model.base(x).square().mean();loss.backward()
        for name,param in model.base.token_trajectory.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
        optimizer.step()
    assert model.base.token_trajectory.gain.abs().sum()>0
    model.eval()
    with torch.no_grad():
        expected=model.base(x);assert torch.equal(expected,model.base(x))
    path=tmp_path/'weights.pth';torch.save(model.state_dict(),path)
    restored=torch.nn.Module();restored.base=backbone(variant,mix).eval()
    build_transformer.load_param(restored,path)
    with torch.no_grad():assert torch.equal(expected,restored.base(x))
    other={'A':'B','B':'A','C':'D','D':'C'}[group]
    wrong=torch.nn.Module();wrong.base=backbone(*CHOICES[other])
    with pytest.raises(ValueError,match='checkpoint/config mismatch'):build_transformer.load_param(wrong,path)


def test_round_three_computation_unchanged_against_previous_commit():
    old=types.ModuleType('model.backbones._before_round4')
    exec(subprocess.check_output(['git','show','eb74e21:model/backbones/vit_pytorch.py'],cwd=ROOT).decode(),old.__dict__)
    choices=[('cls_acceleration',1.),('patch_acceleration',1.),('attention_velocity',0.),('mlp_velocity',0.)]
    for variant,mix in choices:
        torch.manual_seed(31);a=backbone(variant,mix,factory=old.TransReID);rng=torch.random.get_rng_state()
        torch.manual_seed(31);b=backbone(variant,mix)
        assert torch.equal(rng,torch.random.get_rng_state())
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        with torch.no_grad():
            a.token_trajectory.gain.normal_(std=.05);b.token_trajectory.gain.copy_(a.token_trajectory.gain)
        x=torch.randn(6,3,32,16)
        torch.manual_seed(37);y=a(x);rng=torch.random.get_rng_state()
        torch.manual_seed(37);z=b(x)
        assert torch.equal(y,z) and torch.equal(rng,torch.random.get_rng_state())
        y.square().mean().backward();z.square().mean().backward()
        for name,p in a.named_parameters():
            q=dict(b.named_parameters())[name]
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad),name
