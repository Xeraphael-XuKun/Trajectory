"""E1 formula, gradient, recurrence, configuration and reload checks."""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
from model.backbones.token_trajectory_round5 import Round5VelocityTrajectory
from model.backbones.token_trajectory_variants import validate_trajectory_config
from model.backbones.vit_pytorch import TransReID
from model.make_model import build_transformer
from solver.make_optimizer import make_optimizer

ROOT = Path(__file__).resolve().parents[1]


def settings():
    c = cfg.clone()
    c.merge_from_file(str(ROOT / 'configs/trajectory_E1_transient.yml'))
    return c


def tiny(enabled=True, drop_path=0.0):
    return TransReID(img_size=(32, 16), patch_size=16, stride_size=16,
                    embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                    qkv_bias=True, drop_path_rate=drop_path,
                    token_trajectory=enabled,
                    token_trajectory_variant='velocity_transient',
                    token_trajectory_accel_mix=0.0)


def holder(base):
    model = nn.Module()
    model.base = base
    model.bottleneck = nn.BatchNorm1d(8)
    model.classifier = nn.Linear(8, 2, bias=False)
    return model


def test_parameter_shape_signature_and_zero_direction():
    module = Round5VelocityTrajectory(12, 129, 768, 'velocity_transient')
    assert list(dict(module.named_parameters())) == ['gain']
    assert module.gain.shape == (11, 1, 129, 768)
    assert module.trajectory_parameters == 1089792
    assert module.method_signature.tolist() == [701, 1, 12, 129, 768, 0, 0, 1, 0, 1, 1e-5]
    v = torch.ones(2, 129, 768)
    assert torch.equal(module(1, v), torch.zeros_like(v))
    assert torch.isfinite(module.normalized_direction(v)).all()


def test_c0_direction_and_nonzero_velocity_gradient():
    e1 = Round5VelocityTrajectory(4, 5, 8, 'velocity_transient')
    c0 = DenseCrossLayerTokenTrajectory(4, 5, 8, acceleration_mix=0)
    with torch.no_grad():
        e1.gain.normal_()
        c0.gain.copy_(e1.gain)
    v = torch.randn(2, 5, 8, requires_grad=True)
    for i in range(1, 4):
        torch.testing.assert_close(e1(i, v), c0(i, v), rtol=0, atol=0)
    (e1(2, v) * torch.randn_like(v)).sum().backward()
    assert v.grad.abs().sum() > 0


@pytest.mark.parametrize('layer', [0, 4, 1.5])
def test_invalid_layer(layer):
    with pytest.raises(IndexError):
        Round5VelocityTrajectory(4, 5, 8, 'velocity_transient')(
            layer, torch.randn(2, 5, 8))


def test_invalid_history_and_shapes():
    m = Round5VelocityTrajectory(4, 5, 8, 'velocity_transient')
    v = torch.randn(2, 5, 8)
    with pytest.raises(ValueError):
        m(1, v, v)
    with pytest.raises(ValueError):
        m(1, v[:, :4])
    with pytest.raises(ValueError):
        m(1, v, gate=torch.ones(3))


@pytest.mark.parametrize('train', [False, True])
def test_zero_start_preserves_shared_initialization_rng_and_output(train):
    torch.manual_seed(1234)
    base = tiny(enabled=False, drop_path=.2).train(train)
    rng = torch.random.get_rng_state()
    torch.manual_seed(1234)
    e1 = tiny(drop_path=.2).train(train)
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in base.state_dict().items():
        assert torch.equal(v, e1.state_dict()[k]), k
    x = torch.randn(2, 3, 32, 16)
    before = torch.random.get_rng_state()
    expected = base(x)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    assert torch.equal(expected, e1(x))
    assert torch.equal(after, torch.random.get_rng_state())


class LinearBlock(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, chart=None):
        return (1 + self.alpha) * x


class IndependentCorrection(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = nn.Parameter(r)

    def forward(self, layer, velocity, gate=None):
        return self.r if gate is None else self.r * gate[:, None, None]


@pytest.mark.parametrize('alpha', [0.0, .2])
def test_actual_e1_loop_removal_jacobian(alpha):
    model = tiny()
    model.blocks = nn.ModuleList([LinearBlock(.1), LinearBlock(alpha)])
    model.token_trajectory = IndependentCorrection(torch.randn(2, 3, 8))
    h = torch.randn(2, 3, 8)
    direction = torch.randn_like(h)
    y = model._forward_round5_tokens(h)
    r = model.token_trajectory.r
    expected = (1 + alpha) * (1.1 * h) + alpha * r
    torch.testing.assert_close(y, expected, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad((y * direction).sum(), r)[0]
    torch.testing.assert_close(grad, alpha * direction, rtol=1e-6, atol=1e-6)


def test_raw_velocity_cache_final_removal_and_local_history():
    torch.manual_seed(5)
    model = tiny().eval()
    model.token_trajectory.gain.data.normal_(std=.1)
    raw, corrections, observed, final = [], [], [], []
    handles = [b.register_forward_hook(
        lambda m, args, y: raw.append((args[0].clone(), y.clone()))) for b in model.blocks]
    handles.append(model.token_trajectory.register_forward_hook(
        lambda m, args, y: (observed.append((args[0], args[1].clone())),
                            corrections.append(y.clone())) and None))
    handles.append(model.norm.register_forward_pre_hook(
        lambda m, args: final.append(args[0].clone())))
    x = torch.randn(2, 3, 32, 16)
    first = model(x)
    assert [i for i, v in observed] == [1, 2, 3]
    for j, (_, v) in enumerate(observed):
        assert torch.equal(v, raw[j][1] - raw[j][0])
    for j in range(1, 4):
        torch.testing.assert_close(raw[j][0], raw[j-1][1] -
                                   (corrections[j-2] if j > 1 else 0) + corrections[j-1])
    assert torch.equal(final[0], raw[-1][1] - corrections[-1])
    assert torch.equal(first, model(x))
    model(torch.randn_like(x))
    assert torch.equal(first, model(x))
    for h in handles:
        h.remove()


def test_gate_off_and_first_gain_step():
    model = tiny().eval()
    x = torch.randn(2, 3, 32, 16)
    off = model(x, trajectory_gate=torch.zeros(2))
    optimizer = torch.optim.Adam(model.token_trajectory.parameters(), lr=.01)
    loss = (model(x) * torch.randn(2, 8)).sum()
    loss.backward()
    grad = model.token_trajectory.gain.grad
    assert torch.isfinite(grad).all()
    assert (grad.flatten(1).abs().sum(1) > 0).all()
    optimizer.step()
    assert model.token_trajectory.gain.abs().sum() > 0
    assert torch.equal(off, model(x, trajectory_gate=torch.zeros(2)))


def test_recipe_matches_c0_except_declared_fields_and_gain_group():
    e1, c0 = settings(), cfg.clone()
    c0.merge_from_file(str(ROOT / 'configs/trajectory_C0.yml'))
    validate_trajectory_config(e1)
    restored = e1.clone()
    restored.MODEL.TOKEN_TRAJECTORY_VARIANT = c0.MODEL.TOKEN_TRAJECTORY_VARIANT
    restored.TEST.NECK_FEAT = c0.TEST.NECK_FEAT
    restored.OUTPUT_DIR = c0.OUTPUT_DIR
    assert restored.dump() == c0.dump()
    model = holder(tiny())
    optimizer, _ = make_optimizer(e1, model, nn.Linear(1, 1))
    gain = model.base.token_trajectory.gain
    groups = [g for g in optimizer.param_groups if any(p is gain for p in g['params'])]
    assert len(groups) == 1
    assert (groups[0]['lr'], groups[0]['weight_decay']) == (.00035, .0001)
    assert all(p is not model.base.token_trajectory.method_signature
               for g in optimizer.param_groups for p in g['params'])


@pytest.mark.parametrize('owner,key,value', [
    ('MODEL', 'TOKEN_TRAJECTORY', False), ('MODEL', 'TOKEN_TRAJECTORY_ACCEL_MIX', 1.0),
    ('MODEL', 'VPR', True), ('MODEL', 'TEXT_ALIGN', True),
    ('MODEL', 'TOKEN_TRAJECTORY_HIDDEN_DIM', 32),
    ('MODEL', 'PE_FREEZE_BASE', True), ('MODEL', 'CE_SPLIT_MODALITY', True),
    ('MODEL', 'DIST_TRAIN', True), ('DATALOADER', 'SYNC_FRAMES', True),
    ('C0_AUX', 'ENABLED', True), ('TERMINAL_REPAIR', 'ENABLED', True),
    ('SOLVER', 'TNCE_WEIGHT', 1.0),
])
def test_config_rejects_mixed_or_ignored_settings(owner, key, value):
    c = settings()
    setattr(getattr(c, owner), key, value)
    with pytest.raises(ValueError):
        validate_trajectory_config(c)


def test_formal_loader_roundtrip_prefix_and_signature_rejection(tmp_path):
    source, target = holder(tiny()), holder(tiny())
    source.base.token_trajectory.gain.data.normal_(std=.1)
    source.bottleneck.running_mean.normal_()
    source.bottleneck.running_var.uniform_(.5, 1.5)
    source.eval()
    x = torch.randn(2, 3, 32, 16)
    checkpoint = {'module.' + k: v for k, v in source.state_dict().items()}
    path = tmp_path / 'e1.pth'
    torch.save({'state_dict': checkpoint}, path)
    build_transformer.load_param(target, path)
    target.eval()
    assert torch.equal(source.base(x), target.base(x))
    assert torch.equal(source.bottleneck(source.base(x)), target.bottleneck(target.base(x)))
    assert not torch.equal(source.classifier.weight, target.classifier.weight)
    for code in [702, 703]:
        bad = {k: v.clone() for k, v in source.state_dict().items()}
        bad['base.token_trajectory.method_signature'][0] = code
        torch.save(bad, path)
        with pytest.raises(ValueError, match='checkpoint/config mismatch'):
            build_transformer.load_param(target, path)
    old = holder(TransReID(img_size=(32, 16), patch_size=16, stride_size=16,
                          embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                          token_trajectory=True, token_trajectory_accel_mix=0))
    torch.save(old.state_dict(), path)
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        build_transformer.load_param(target, path)
    torch.save(source.state_dict(), path)
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        build_transformer.load_param(old, path)


@pytest.mark.parametrize('key', ['base.token_trajectory.gain', 'bottleneck.running_mean',
                                 'bottleneck.running_var', 'bottleneck.num_batches_tracked'])
def test_missing_retrieval_state_rejected_before_copy(tmp_path, key):
    source, target = holder(tiny()), holder(tiny())
    before = {k: v.clone() for k, v in target.state_dict().items()}
    state = source.state_dict()
    state.pop(key)
    path = tmp_path / 'missing.pth'
    torch.save(state, path)
    with pytest.raises(ValueError, match='Incomplete E1 checkpoint'):
        build_transformer.load_param(target, path)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())
