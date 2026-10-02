"""Checks for token routing and reuse of the exact realized block residuals."""
from pathlib import Path
import subprocess
import types

import pytest
import torch
import torch.nn.functional as F

from config import cfg
from model.backbones.vit_pytorch import Block, TransReID
from model.backbones.token_trajectory_structure import StructuredTokenTrajectory
from model.backbones.token_trajectory_variants import validate_trajectory_config
from model.make_model import build_transformer
from solver import make_optimizer

ROOT = Path(__file__).resolve().parents[1]
CHOICES = {'P1': ('cls_acceleration', 1.0), 'P2': ('patch_acceleration', 1.0),
           'P3': ('attention_velocity', 0.0), 'P4': ('mlp_velocity', 0.0)}


def backbone(variant='dense', mix=1.0, enabled=True, factory=TransReID):
    return factory(img_size=(32, 16), patch_size=16, stride_size=16,
                   embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                   drop_path_rate=0.2, token_trajectory=enabled,
                   token_trajectory_variant=variant, token_trajectory_accel_mix=mix)


def old_module():
    old = types.ModuleType('model.backbones._before_structure')
    exec(subprocess.check_output(['git', 'show', 'a391b06:model/backbones/vit_pytorch.py'],
                                 cwd=ROOT).decode(), old.__dict__)
    return old


@pytest.mark.parametrize('layer_scale', [False, True])
def test_block_residual_capture_reuses_dropout_draws_and_preserves_gradients(layer_scale):
    kwargs = dict(dim=8, num_heads=2, mlp_ratio=2, drop=0.15, attn_drop=0.1,
                  drop_path=0.3, layer_scale=layer_scale)
    torch.manual_seed(5)
    old = old_module().Block(**kwargs).train()
    torch.manual_seed(5)
    new = Block(**kwargs).train()
    x = torch.randn(6, 3, 8)
    seen = []
    handle = new.drop_path.register_forward_hook(lambda m, inputs, out: seen.append(out))
    torch.manual_seed(27)
    expected = old(x)
    old_rng = torch.random.get_rng_state()
    expected.square().mean().backward()
    torch.manual_seed(27)
    actual, attention, mlp = new(x, return_updates=True)
    handle.remove()
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert torch.equal(expected, actual)
    assert len(seen) == 2 and attention is seen[0] and mlp is seen[1]
    assert torch.equal(actual, (x + attention) + mlp)
    actual.square().mean().backward()
    for name, p in old.named_parameters():
        assert torch.equal(p.grad, dict(new.named_parameters())[name].grad), name


@pytest.mark.parametrize('variant,cls_beta,patch_beta', [
    ('cls_acceleration', 1, 0), ('patch_acceleration', 0, 1)])
def test_role_formula_uses_only_selected_tokens(variant, cls_beta, patch_beta):
    module = StructuredTokenTrajectory(4, 3, 8, variant)
    v, old = torch.randn(2, 3, 8), torch.randn(2, 3, 8, requires_grad=True)
    with torch.no_grad():
        module.gain.normal_()
    beta = torch.tensor([cls_beta, patch_beta, patch_beta]).reshape(1, 3, 1)
    expected = module.gain[1] * F.layer_norm(v + beta * (v - old), (8,))
    assert torch.equal(module(2, v, old), expected)
    module(2, v, old).square().sum().backward()
    inactive = old.grad[:, :1] if cls_beta == 0 else old.grad[:, 1:]
    active = old.grad[:, :1] if cls_beta else old.grad[:, 1:]
    assert torch.count_nonzero(inactive) == 0
    assert torch.count_nonzero(active) > 0
    assert torch.equal(module(1, v), module.gain[0] * F.layer_norm(v, (8,)))


@pytest.mark.parametrize('group', CHOICES)
def test_real_configuration_source_routing_learning_and_checkpoint(group, tmp_path):
    c = cfg.clone()
    c.merge_from_file(str(ROOT / ('configs/trajectory_' + group + '.yml')))
    validate_trajectory_config(c)
    variant, mix = CHOICES[group]
    assert (c.MODEL.TOKEN_TRAJECTORY_VARIANT, c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX) == (variant, mix)
    assert c.TEST.NECK_FEAT == 'after'
    bad = c.clone(); bad.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX = 0.5
    with pytest.raises(ValueError):
        validate_trajectory_config(bad)
    torch.manual_seed(1234)
    reference = backbone(enabled=False)
    rng = torch.random.get_rng_state()
    torch.manual_seed(1234)
    model = torch.nn.Module(); model.base = backbone(variant, mix)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, model.base.state_dict()[key]), key
    x, target = torch.randn(6, 3, 32, 16), torch.randn(6, 8)
    torch.manual_seed(17); expected = reference(x)
    rng = torch.random.get_rng_state()
    torch.manual_seed(17); actual = model.base(x)
    assert torch.equal(expected, actual) and torch.equal(rng, torch.random.get_rng_state())
    optimizer, _ = make_optimizer(c, model, torch.nn.Linear(8, 2))
    for pg in optimizer.param_groups:
        assert pg['weight_decay'] == 1e-4
    realized, observations, hooks = [], [], []
    for block in model.base.blocks:
        hooks.append(block.drop_path.register_forward_hook(
            lambda m, inputs, out: realized.append(out.detach().clone())))
    hooks.append(model.base.token_trajectory.register_forward_pre_hook(
        lambda m, args: observations.append((args[0], args[1].detach().clone()))))
    model.base(x)
    for h in hooks: h.remove()
    assert len(realized) == 8 and [i for i, v in observations] == [1, 2, 3]
    if group in ('P3', 'P4'):
        offset = 0 if group == 'P3' else 1
        for i, velocity in observations:
            assert torch.equal(velocity, realized[2 * (i - 1) + offset])
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model.base(x) - target).square().mean()
        loss.backward()
        grad = model.base.token_trajectory.gain.grad
        assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.abs().sum() > 0
        optimizer.step()
    model.eval()
    with torch.no_grad():
        first = model.base(x)
        assert torch.equal(first, model.base(x))  # no history retained across inputs
    path = tmp_path / 'checkpoint.pth'
    torch.save(model.state_dict(), path)
    restored = torch.nn.Module(); restored.base = backbone(variant, mix).eval()
    build_transformer.load_param(restored, path)
    with torch.no_grad(): assert torch.equal(first, restored.base(x))
    other_variant, other_mix = CHOICES['P2' if group != 'P2' else 'P1']
    wrong = torch.nn.Module(); wrong.base = backbone(other_variant, other_mix)
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        build_transformer.load_param(wrong, path)


def test_all_old_backbones_preserve_training_outputs_rng_and_gradients():
    old = old_module()
    choices = [('baseline', 'dense', 1.0, False)]
    for group in ('T0', 'C0', 'T1', 'T2', 'T3', 'T4', 'T5', 'R1', 'R2', 'R3', 'R4'):
        c = cfg.clone(); c.merge_from_file(str(ROOT / ('configs/trajectory_' + group + '.yml')))
        choices.append((group, c.MODEL.TOKEN_TRAJECTORY_VARIANT, c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX, True))
    for group, variant, mix, enabled in choices:
        torch.manual_seed(12); a = backbone(variant, mix, enabled, old.TransReID)
        rng = torch.random.get_rng_state()
        torch.manual_seed(12); b = backbone(variant, mix, enabled)
        assert torch.equal(rng, torch.random.get_rng_state()), group
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()), group
        if enabled:
            with torch.no_grad():
                for p in a.token_trajectory.parameters(): p.normal_(std=0.05)
                for n, p in b.token_trajectory.named_parameters():
                    p.copy_(dict(a.token_trajectory.named_parameters())[n])
        x = torch.randn(6, 3, 32, 16)
        torch.manual_seed(33); y = a(x); rng = torch.random.get_rng_state()
        torch.manual_seed(33); z = b(x)
        assert torch.equal(y, z) and torch.equal(rng, torch.random.get_rng_state()), group
        y.square().mean().backward(); z.square().mean().backward()
        for n, p in a.named_parameters():
            q = dict(b.named_parameters())[n]
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad), (group, n)
