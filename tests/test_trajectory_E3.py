"""E3 formula, gradient, recurrence, configuration and reload checks."""
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
    c.merge_from_file(str(ROOT / 'configs/trajectory_E3_uniform_transport.yml'))
    return c


def tiny(enabled=True, drop_path=0.0):
    return TransReID(img_size=(32, 16), patch_size=16, stride_size=16,
                    embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                    qkv_bias=True, drop_path_rate=drop_path,
                    token_trajectory=enabled,
                    token_trajectory_variant='velocity_uniform_transport',
                    token_trajectory_accel_mix=0.0)


def holder(base):
    model = nn.Module()
    model.base = base
    model.bottleneck = nn.BatchNorm1d(8)
    model.classifier = nn.Linear(8, 2, bias=False)
    return model


def test_parameter_shape_signature_and_zero_direction():
    module = Round5VelocityTrajectory(12, 129, 768, 'velocity_uniform_transport')
    assert list(dict(module.named_parameters())) == ['gain']
    assert module.gain.shape == (11, 1, 129, 768)
    assert module.trajectory_parameters == 1089792
    assert module.method_signature.tolist() == [703, 1, 12, 129, 768, 0, .5, 0, 0, 1, 1e-5]
    v = torch.ones(2, 129, 768)
    assert torch.equal(module(1, v), torch.zeros_like(v))
    assert torch.isfinite(module.normalized_direction(v)).all()


def test_per_image_token_mean_with_cls_and_uniform_matrix_control():
    torch.manual_seed(3)
    e3 = Round5VelocityTrajectory(4, 5, 8, 'velocity_uniform_transport')
    e3.gain.data.normal_()
    v = torch.randn(2, 5, 8)
    expected = e3.gain[0] * F.layer_norm(.5*v + .5*v.mean(1, keepdim=True), (8,))
    actual = e3(1, v)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # E2 with an artificial uniform P: same full velocity, mixture and gain.
    uniform = torch.full((2, 5, 5), 1/5)
    e2_uniform = e3.gain[0] * F.layer_norm(.5*v + .5*torch.bmm(uniform, v), (8,))
    torch.testing.assert_close(actual, e2_uniform, rtol=1e-6, atol=1e-6)
    wrong_order = e3.gain[0] * (.5*F.layer_norm(v, (8,)) +
                              .5*F.layer_norm(v.mean(1, keepdim=True), (8,)))
    assert not torch.allclose(actual, wrong_order)
    other = v.clone()
    other[1] += 9*torch.randn_like(other[1])
    assert torch.equal(actual[0], e3(1, other)[0])
    cls_changed = v.clone()
    cls_changed[:, 0] += torch.arange(8)
    assert not torch.allclose(actual[:, 1:], e3(1, cls_changed)[:, 1:])
    # Equal token velocities make the mixture exactly the C0 direction.
    same = torch.randn(2, 1, 8).expand(-1, 5, -1)
    c0 = DenseCrossLayerTokenTrajectory(4, 5, 8, acceleration_mix=0)
    c0.gain.data.copy_(e3.gain)
    torch.testing.assert_close(e3(1, same), c0(1, same), rtol=1e-6, atol=1e-6)


def test_mean_path_gradient_is_not_detached():
    torch.manual_seed(7)
    e3 = Round5VelocityTrajectory(4, 5, 8, 'velocity_uniform_transport')
    e3.gain.data.normal_()
    v = torch.randn(2, 5, 8, requires_grad=True)
    # Only patch1 contributes to loss; all other tokens must receive mean-path gradients.
    weights = torch.zeros_like(v)
    weights[:, 1] = torch.randn(2, 8)
    (e3(1, v)*weights).sum().backward()
    reference = v.detach().clone().requires_grad_()
    mixed = .5*reference + .5*reference.mean(1, keepdim=True)
    (e3.gain[0].detach()*F.layer_norm(mixed, (8,))*weights).sum().backward()
    torch.testing.assert_close(v.grad, reference.grad, rtol=0, atol=0)
    assert (v.grad[:, [0, 2, 3, 4]].abs().sum(-1) > 0).all()
    detached = v.detach().clone().requires_grad_()
    wrong = .5*detached + .5*detached.mean(1, keepdim=True).detach()
    (e3.gain[0].detach()*F.layer_norm(wrong, (8,))*weights).sum().backward()
    assert detached.grad[:, [0, 2, 3, 4]].abs().sum() == 0
    assert not torch.allclose(v.grad, detached.grad)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_mean_and_ln_fp32_inside_autocast_without_dense_matrix(dtype, monkeypatch):
    m = Round5VelocityTrajectory(4, 5, 8, 'velocity_uniform_transport')
    m.gain.data.normal_()
    v = torch.randn(2, 5, 8).to(dtype)
    mixed = .5*v.float()+.5*v.float().mean(1, keepdim=True)
    expected = F.layer_norm(mixed, (8,)).to(dtype)*m.gain[0].to(dtype)
    original, calls = F.layer_norm, []
    def checked(x, *args, **kwargs):
        calls.append((x.dtype, torch.is_autocast_cpu_enabled()))
        assert torch.equal(x, mixed)
        return original(x, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('E3 must use a token mean, without dense uniform bmm')
    monkeypatch.setattr(F, 'layer_norm', checked)
    monkeypatch.setattr(torch, 'bmm', forbidden)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        actual = m(1, v)
    assert calls == [(torch.float32, False)]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('layer', [0, 4, 1.5])
def test_invalid_layer(layer):
    with pytest.raises(IndexError):
        Round5VelocityTrajectory(4, 5, 8, 'velocity_uniform_transport')(
            layer, torch.randn(2, 5, 8))


def test_invalid_history_and_shapes():
    m = Round5VelocityTrajectory(4, 5, 8, 'velocity_uniform_transport')
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
    e3 = tiny(drop_path=.2).train(train)
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in base.state_dict().items():
        assert torch.equal(v, e3.state_dict()[k]), k
    x = torch.randn(2, 3, 32, 16)
    before = torch.random.get_rng_state()
    expected = base(x)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    assert torch.equal(expected, e3(x))
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
def test_actual_e3_loop_persistent_jacobian(alpha):
    model = tiny()
    model.blocks = nn.ModuleList([LinearBlock(.1), LinearBlock(alpha)])
    model.token_trajectory = IndependentCorrection(torch.randn(2, 3, 8))
    h = torch.randn(2, 3, 8)
    direction = torch.randn_like(h)
    y = model._forward_round5_tokens(h)
    r = model.token_trajectory.r
    expected = (1 + alpha) * (1.1 * h + r)
    torch.testing.assert_close(y, expected, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad((y * direction).sum(), r)[0]
    torch.testing.assert_close(grad, (1 + alpha) * direction, rtol=1e-6, atol=1e-6)


def test_raw_velocity_cache_persistence_no_attention_and_local_history():
    torch.manual_seed(5)
    model = tiny().eval()
    model.token_trajectory.gain.data.normal_(std=.1)
    raw, corrections, observed, final, flags = [], [], [], [], []
    handles = [b.register_forward_hook(
        lambda m, args, y: raw.append((args[0].clone(), y.clone()))) for b in model.blocks]
    handles.extend(b.register_forward_pre_hook(
        lambda m, args, kwargs: flags.append(kwargs.get('return_mean_attention', False)),
        with_kwargs=True) for b in model.blocks)
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
        assert torch.equal(raw[j][0], raw[j-1][1] + corrections[j-1])
    assert torch.equal(final[0], raw[-1][1])
    assert flags == [False]*4
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
    e3, c0 = settings(), cfg.clone()
    c0.merge_from_file(str(ROOT / 'configs/trajectory_C0.yml'))
    validate_trajectory_config(e3)
    restored = e3.clone()
    restored.MODEL.TOKEN_TRAJECTORY_VARIANT = c0.MODEL.TOKEN_TRAJECTORY_VARIANT
    restored.TEST.NECK_FEAT = c0.TEST.NECK_FEAT
    restored.OUTPUT_DIR = c0.OUTPUT_DIR
    assert restored.dump() == c0.dump()
    model = holder(tiny())
    optimizer, _ = make_optimizer(e3, model, nn.Linear(1, 1))
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
    path = tmp_path / 'e3.pth'
    torch.save({'state_dict': checkpoint}, path)
    build_transformer.load_param(target, path)
    target.eval()
    assert torch.equal(source.base(x), target.base(x))
    assert torch.equal(source.bottleneck(source.base(x)), target.bottleneck(target.base(x)))
    assert not torch.equal(source.classifier.weight, target.classifier.weight)
    for code in [701, 702]:
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
    with pytest.raises(ValueError, match='Incomplete E3 checkpoint'):
        build_transformer.load_param(target, path)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())
