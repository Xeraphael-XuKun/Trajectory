"""Numerical checks for round-two method isolation and actual optimizer routing."""
from pathlib import Path
import subprocess
import types

import pytest
import torch
import torch.nn.functional as F

from config import cfg
from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
from model.backbones.token_trajectory_variants import TrajectoryVariant, validate_trajectory_config
from model.backbones.token_trajectory_refinements import TrajectoryRefinement
from model.backbones.vit_pytorch import TransReID
from model.make_model import build_transformer
from solver import make_optimizer

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'R1': ('dense_half', 0.5), 'R2': ('adaptive_half', 1.0),
          'R3': ('token_gate_no_decay', 1.0), 'R4': ('velocity_gate', 0.0)}


def configuration(group):
    result = cfg.clone()
    result.merge_from_file(str(ROOT / ('configs/trajectory_' + group + '.yml')))
    validate_trajectory_config(result)
    return result


def tiny(variant='dense', mix=1.0, enabled=True):
    return TransReID(img_size=(32, 16), patch_size=16, stride_size=16,
                    embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                    token_trajectory=enabled, token_trajectory_variant=variant,
                    token_trajectory_accel_mix=mix)


@pytest.mark.parametrize('group', GROUPS)
def test_configuration_initialization_learning_and_optimizer(group, tmp_path):
    c = configuration(group)
    variant, mix = GROUPS[group]
    assert c.MODEL.TOKEN_TRAJECTORY_VARIANT == variant
    assert c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX == mix
    assert c.TEST.NECK_FEAT == 'after'
    torch.manual_seed(1234)
    reference = tiny(enabled=False).eval()
    rng = torch.random.get_rng_state()
    torch.manual_seed(1234)
    model = torch.nn.Module()
    model.base = tiny(variant, mix).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, model.base.state_dict()[key]), key
    x = torch.randn(3, 3, 32, 16)
    with torch.no_grad():
        assert torch.equal(reference(x), model.base(x))
    center = torch.nn.Linear(8, 2)
    optimizer, _ = make_optimizer(c, model, center)
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    for pg in optimizer.param_groups:
        name = name_by_id[id(pg['params'][0])]
        gate = '.conditioners.' in name and group in ('R3', 'R4')
        assert pg['weight_decay'] == (0.0 if gate else 1e-4), name
        assert pg['lr'] == (0.00035 if '.token_trajectory.' in name else 0.000005), name
    target = torch.randn(3, 8)
    reached = set()
    for _ in range(5):
        optimizer.zero_grad()
        loss = (model.base(x) - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in model.base.token_trajectory.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if p.grad.abs().sum() > 0:
                reached.add(name)
        optimizer.step()
    assert reached == set(dict(model.base.token_trajectory.named_parameters()))
    path = tmp_path / 'model.pth'
    torch.save(model.state_dict(), path)
    restored = torch.nn.Module()
    restored.base = tiny(variant, mix).eval()
    build_transformer.load_param(restored, path)
    with torch.no_grad():
        assert torch.equal(model.base(x), restored.base(x))
    wrong = torch.nn.Module()
    wrong.base = tiny().eval()
    before = wrong.base.token_trajectory.gain.clone()
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        build_transformer.load_param(wrong, path)
    assert torch.equal(before, wrong.base.token_trajectory.gain)


def test_half_formulas_and_velocity_gate_has_no_acceleration_path():
    v, old = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    for variant, mix in (('dense_half', 0.5), ('adaptive_half', 1.0)):
        m = TrajectoryRefinement(4, 3, 8, variant, mix)
        with torch.no_grad():
            m.gain.normal_()
        expected = m.gain[1] * F.layer_norm(v + 0.5 * (v - old), (8,))
        assert torch.allclose(m(2, v, old), expected, atol=1e-6)
        if variant == 'adaptive_half':
            assert torch.allclose(m.adaptive_beta(2, v, old), torch.full((2, 3, 1), 0.5))
    m = TrajectoryRefinement(4, 3, 8, 'velocity_gate', 0.0)
    with torch.no_grad():
        m.gain.normal_()
    assert torch.equal(m(2, v, old), m.gain[1] * F.layer_norm(v, (8,)))
    with torch.no_grad():
        for p in m.conditioners.parameters():
            p.normal_(std=0.2)
    old.requires_grad_()
    result = m(2, v, old)
    assert torch.equal(result, m(2, v, torch.randn_like(old)))
    result.sum().backward()
    assert old.grad is None


def test_r3_has_identical_t4_function_and_only_gate_decay_changes():
    torch.manual_seed(12)
    t4 = TrajectoryVariant(4, 3, 8, 'token_gate')
    torch.manual_seed(12)
    r3 = TrajectoryRefinement(4, 3, 8, 'token_gate_no_decay')
    for name, p in t4.named_parameters():
        assert torch.equal(p, dict(r3.named_parameters())[name])
    with torch.no_grad():
        for p in t4.parameters():
            p.normal_(std=0.1)
        for name, p in r3.named_parameters():
            p.copy_(dict(t4.named_parameters())[name])
    v, old = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    assert torch.equal(t4(2, v, old), r3(2, v, old))
    t4(2, v, old).sum().backward()
    r3(2, v, old).sum().backward()
    assert all(torch.equal(p.grad, dict(r3.named_parameters())[n].grad)
               for n, p in t4.named_parameters() if p.grad is not None)


def test_historical_variant_outputs_gradients_and_optimizer_unchanged():
    old = types.ModuleType('model.backbones._old_variants')
    exec(subprocess.check_output(['git', 'show', 'a8349ab:model/backbones/token_trajectory_variants.py'],
                                 cwd=ROOT).decode(), old.__dict__)
    old_opt = types.ModuleType('_old_optimizer')
    exec(subprocess.check_output(['git', 'show', 'a8349ab:solver/make_optimizer.py'],
                                 cwd=ROOT).decode(), old_opt.__dict__)
    for variant in ('adaptive', 'split', 'ema', 'token_gate', 'channel_mix'):
        torch.manual_seed(10)
        a = old.TrajectoryVariant(4, 3, 8, variant)
        rng = torch.random.get_rng_state()
        torch.manual_seed(10)
        b = TrajectoryVariant(4, 3, 8, variant)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
        with torch.no_grad():
            for p in a.parameters():
                p.normal_(std=0.1)
            for n, p in b.named_parameters():
                p.copy_(dict(a.named_parameters())[n])
        v, old_v = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
        assert torch.equal(a(2, v, old_v), b(2, v, old_v))
        a(2, v, old_v).sum().backward()
        b(2, v, old_v).sum().backward()
        for n, p in a.named_parameters():
            q = dict(b.named_parameters())[n]
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)
    for group in ('T0', 'C0', 'T1', 'T2', 'T3', 'T4', 'T5'):
        c = configuration(group)
        model = torch.nn.Module()
        model.base = tiny(c.MODEL.TOKEN_TRAJECTORY_VARIANT, c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX)
        center = torch.nn.Linear(8, 2)
        a, _ = old_opt.make_optimizer(c, model, center)
        b, _ = make_optimizer(c, model, center)
        assert a.state_dict()['param_groups'] == b.state_dict()['param_groups']
