"""E2 transport formula, gradient, causal recurrence, configuration and reload checks."""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
from model.backbones.token_trajectory_round5 import Round5VelocityTrajectory
from model.backbones.token_trajectory_variants import validate_trajectory_config
from model.backbones.vit_pytorch import Attention, Block, TransReID
from model.make_model import build_transformer
from solver.make_optimizer import make_optimizer

ROOT = Path(__file__).resolve().parents[1]


def settings():
    c = cfg.clone()
    c.merge_from_file(str(ROOT / 'configs/trajectory_E2_attn_transport.yml'))
    return c


def tiny(enabled=True, drop_path=0.0):
    return TransReID(img_size=(32, 16), patch_size=16, stride_size=16,
                    embed_dim=8, depth=4, num_heads=2, mlp_ratio=2,
                    qkv_bias=True, drop_path_rate=drop_path,
                    token_trajectory=enabled,
                    token_trajectory_variant='velocity_attn_transport',
                    token_trajectory_accel_mix=0.0)


def holder(base):
    model = nn.Module()
    model.base = base
    model.bottleneck = nn.BatchNorm1d(8)
    model.classifier = nn.Linear(8, 2, bias=False)
    return model


def test_parameter_shape_signature_and_zero_direction():
    module = Round5VelocityTrajectory(12, 129, 768, 'velocity_attn_transport')
    assert list(dict(module.named_parameters())) == ['gain']
    assert module.gain.shape == (11, 1, 129, 768)
    assert module.trajectory_parameters == 1089792
    assert module.method_signature.tolist() == [702, 1, 12, 129, 768, 0, .5, 0, 1, 1, 1e-5]
    v = torch.ones(2, 129, 768)
    assert torch.equal(module(1, v, previous_attention=torch.eye(129).expand(2, -1, -1)), torch.zeros_like(v))
    assert torch.isfinite(module.normalized_direction(v, torch.eye(129).expand(2, -1, -1))).all()


def test_route_orientation_ln_order_identity_and_uniform_controls():
    torch.manual_seed(3)
    e2 = Round5VelocityTrajectory(4, 3, 8, 'velocity_attn_transport')
    e2.gain.data.normal_()
    v = torch.randn(2, 3, 8)
    p = torch.tensor([[[.1, .8, .1], [.2, .3, .5], [.7, .1, .2]],
                      [[.9, .05, .05], [.1, .2, .7], [.4, .5, .1]]])
    expected = e2.gain[0] * F.layer_norm(.5*v + .5*torch.bmm(p, v), (8,))
    actual = e2(1, v, previous_attention=p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reversed_route = e2.gain[0] * F.layer_norm(.5*v + .5*torch.bmm(p.transpose(1, 2), v), (8,))
    assert not torch.allclose(actual, reversed_route)
    wrong_order = e2.gain[0] * (.5*F.layer_norm(v, (8,)) +
                              .5*torch.bmm(p, F.layer_norm(v, (8,))))
    assert not torch.allclose(actual, wrong_order)
    c0 = DenseCrossLayerTokenTrajectory(4, 3, 8, acceleration_mix=0)
    c0.gain.data.copy_(e2.gain)
    torch.testing.assert_close(e2(1, v, previous_attention=torch.eye(3).expand(2, -1, -1)),
                               c0(1, v), rtol=0, atol=0)
    uniform = torch.full((2, 3, 3), 1/3)
    # E3 formula control only; E3 is not registered or implemented in this branch.
    e3 = e2.gain[0] * F.layer_norm(.5*v + .5*v.mean(1, keepdim=True), (8,))
    torch.testing.assert_close(e2(1, v, previous_attention=uniform), e3, rtol=1e-6, atol=1e-6)
    changed = v.clone()
    changed[1] += 7*torch.randn_like(changed[1])
    assert torch.equal(actual[0], e2(1, changed, previous_attention=p)[0])
    # CLS is a normal source and receiver: routing from token0 affects patches.
    cls_route = torch.zeros_like(p)
    cls_route[:, :, 0] = 1
    changed = v.clone()
    changed[:, 0] += torch.arange(8)
    assert not torch.allclose(e2(1, v, previous_attention=cls_route)[:, 1:],
                              e2(1, changed, previous_attention=cls_route)[:, 1:])


def test_stop_gradient_only_for_p_and_exact_velocity_gradient():
    torch.manual_seed(7)
    e2 = Round5VelocityTrajectory(4, 3, 8, 'velocity_attn_transport')
    e2.gain.data.normal_()
    v = torch.randn(2, 3, 8, requires_grad=True)
    p = torch.randn(2, 3, 3).softmax(-1).requires_grad_()
    weights = torch.randn_like(v)
    (e2(1, v, previous_attention=p)*weights).sum().backward()
    reference = v.detach().clone().requires_grad_()
    mixed = .5*reference + .5*torch.bmm(p.detach(), reference)
    (e2.gain[0].detach()*F.layer_norm(mixed, (8,))*weights).sum().backward()
    assert p.grad is None
    assert v.grad.abs().sum() > 0
    torch.testing.assert_close(v.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_fp32_transport_inside_autocast_and_actual_velocity_dtype(dtype, monkeypatch):
    m = Round5VelocityTrajectory(4, 3, 8, 'velocity_attn_transport')
    m.gain.data.normal_()
    v = torch.randn(2, 3, 8).to(dtype)
    p = torch.randn(2, 3, 3).softmax(-1).to(torch.bfloat16)
    expected = (F.layer_norm(.5*v.float()+.5*torch.bmm(p.float(), v.float()), (8,))
                .to(dtype)*m.gain[0].to(dtype))
    original, calls = torch.bmm, []
    def checked(a, b):
        calls.append((a.dtype, b.dtype, torch.is_autocast_cpu_enabled()))
        return original(a, b)
    monkeypatch.setattr(torch, 'bmm', checked)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        actual = m(1, v, previous_attention=p)
    assert calls == [(torch.float32, torch.float32, False)]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('layer', [0, 4, 1.5])
def test_invalid_layer(layer):
    with pytest.raises(IndexError):
        Round5VelocityTrajectory(4, 5, 8, 'velocity_attn_transport')(
            layer, torch.randn(2, 5, 8))


def test_invalid_history_and_shapes():
    m = Round5VelocityTrajectory(4, 5, 8, 'velocity_attn_transport')
    v = torch.randn(2, 5, 8)
    with pytest.raises(ValueError, match='needs previous_attention'):
        m(1, v)
    with pytest.raises(ValueError, match='Attention must'):
        m(1, v, previous_attention=torch.eye(4).expand(2, -1, -1))
    with pytest.raises(ValueError):
        m(1, v, v)
    with pytest.raises(ValueError):
        m(1, v[:, :4])
    with pytest.raises(ValueError):
        m(1, v, gate=torch.ones(3), previous_attention=torch.eye(5).expand(2, -1, -1))


@pytest.mark.parametrize('train', [False, True])
def test_zero_start_preserves_shared_initialization_rng_and_output(train):
    torch.manual_seed(1234)
    base = tiny(enabled=False, drop_path=.2).train(train)
    rng = torch.random.get_rng_state()
    torch.manual_seed(1234)
    e2 = tiny(drop_path=.2).train(train)
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in base.state_dict().items():
        assert torch.equal(v, e2.state_dict()[k]), k
    x = torch.randn(2, 3, 32, 16)
    before = torch.random.get_rng_state()
    expected = base(x)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    assert torch.equal(expected, e2(x))
    assert torch.equal(after, torch.random.get_rng_state())


def test_causal_full_velocity_persistent_cache_and_single_attention_calls():
    torch.manual_seed(5)
    model = tiny().eval()
    model.token_trajectory.gain.data.normal_(std=.1)
    raw, corrections, observed, final, flags = [], [], [], [], []
    counts = [0]*4
    handles = []
    for j, block in enumerate(model.blocks):
        def capture(module, args, kwargs, output, j=j):
            flags.append(kwargs.get('return_mean_attention', False))
            y, p = output if isinstance(output, tuple) else (output, None)
            raw.append((args[0].clone(), y.clone(), p))
        def qkv(module, args, output, j=j):
            counts[j] += 1
        handles.append(block.register_forward_hook(capture, with_kwargs=True))
        handles.append(block.attn.qkv.register_forward_hook(qkv))
    def observe(module, args, kwargs, output):
        observed.append((args[0], args[1].clone(), kwargs['previous_attention']))
        corrections.append(output.clone())
    handles.append(model.token_trajectory.register_forward_hook(observe, with_kwargs=True))
    handles.append(model.norm.register_forward_pre_hook(lambda m, args: final.append(args[0].clone())))
    x = torch.randn(2, 3, 32, 16)
    first = model(x)
    assert flags == [True, True, True, False] and counts == [1]*4
    assert [i for i, v, p in observed] == [1, 2, 3]
    for j, (_, v, p) in enumerate(observed):
        assert torch.equal(v, raw[j][1] - raw[j][0])
        assert p is raw[j][2] and not p.requires_grad
    for j in range(1, 4):
        assert torch.equal(raw[j][0], raw[j-1][1] + corrections[j-1])
    assert torch.equal(final[0], raw[-1][1])
    for h in handles:
        h.remove()
    assert torch.equal(first, model(x))
    model(torch.randn_like(x))
    assert torch.equal(first, model(x))


@pytest.mark.parametrize('kind', ['attention', 'block', 'block_rope_scale'])
@pytest.mark.parametrize('train', [False, True])
def test_optional_attention_output_preserves_original_forward_rng_gradients(kind, train):
    torch.manual_seed(13)
    if kind == 'attention':
        module = Attention(8, num_heads=2, qkv_bias=True, attn_drop=.3, proj_drop=.2)
        chart = None
    else:
        extra = {'rope': True, 'layer_scale': True} if kind == 'block_rope_scale' else {}
        module = Block(8, 2, mlp_ratio=2, qkv_bias=True, drop=.2, attn_drop=.3,
                       drop_path=.4, **extra)
        chart = torch.cat([torch.zeros(2, 1, 2), torch.randn(2, 2, 2)], dim=1) if extra else None
    module.train(train)
    x = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn_like(x)
    rng = torch.random.get_rng_state()
    normal = module(x, chart=chart)
    normal_rng = torch.random.get_rng_state()
    (normal*weight).sum().backward()
    grads = {k: None if p.grad is None else p.grad.clone() for k, p in module.named_parameters()}
    dx = x.grad.clone()
    module.zero_grad()
    x.grad = None
    torch.random.set_rng_state(rng)
    output, p = module(x, chart=chart, return_mean_attention=True)
    assert torch.equal(normal, output) and torch.equal(normal_rng, torch.random.get_rng_state())
    assert p.shape == (2, 3, 3) and p.dtype == torch.float32 and not p.requires_grad
    torch.testing.assert_close(p.sum(-1), torch.ones(2, 3), rtol=1e-6, atol=1e-6)
    (output*weight).sum().backward()
    assert torch.equal(dx, x.grad)
    for key, parameter in module.named_parameters():
        assert ((parameter.grad is None and grads[key] is None) or
                torch.equal(parameter.grad, grads[key])), key
    assert module.qkv.weight.grad.abs().sum() > 0 if kind == 'attention' else module.attn.qkv.weight.grad.abs().sum() > 0
    if kind == 'block':
        torch.random.set_rng_state(rng)
        updates, a, m = module(x, return_updates=True)
        assert torch.equal(updates, normal)
        torch.testing.assert_close(updates, x+a+m, rtol=0, atol=0)
        with pytest.raises(ValueError):
            module(x, return_updates=True, return_mean_attention=True)


def test_mean_attention_exactly_pre_dropout_probabilities():
    module = Attention(8, num_heads=2, qkv_bias=True, attn_drop=.9).train()
    x = torch.randn(2, 3, 8)
    q, k, v = module.qkv(x).reshape(2, 3, 3, 2, 4).permute(2, 0, 3, 1, 4)
    expected = ((q@k.transpose(-2, -1))*module.scale).softmax(-1).detach().mean(1)
    _, actual = module(x, return_mean_attention=True)
    assert torch.equal(actual, expected)


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
    e2, c0 = settings(), cfg.clone()
    c0.merge_from_file(str(ROOT / 'configs/trajectory_C0.yml'))
    validate_trajectory_config(e2)
    restored = e2.clone()
    restored.MODEL.TOKEN_TRAJECTORY_VARIANT = c0.MODEL.TOKEN_TRAJECTORY_VARIANT
    restored.TEST.NECK_FEAT = c0.TEST.NECK_FEAT
    restored.OUTPUT_DIR = c0.OUTPUT_DIR
    assert restored.dump() == c0.dump()
    model = holder(tiny())
    optimizer, _ = make_optimizer(e2, model, nn.Linear(1, 1))
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
    path = tmp_path / 'e2.pth'
    torch.save({'state_dict': checkpoint}, path)
    build_transformer.load_param(target, path)
    target.eval()
    assert torch.equal(source.base(x), target.base(x))
    assert torch.equal(source.bottleneck(source.base(x)), target.bottleneck(target.base(x)))
    assert not torch.equal(source.classifier.weight, target.classifier.weight)
    for code in [701, 703]:
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
    with pytest.raises(ValueError, match='Incomplete E2 checkpoint'):
        build_transformer.load_param(target, path)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())
