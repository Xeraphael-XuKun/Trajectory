"""Small numerical checks for the five independent Trajectory experiments."""
import io

import pytest
import torch

from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
from model.backbones.token_trajectory_variants import TrajectoryVariant
from model.backbones.vit_pytorch import TransReID
from model.make_model import build_transformer


VARIANTS = ['adaptive', 'split', 'ema', 'token_gate', 'channel_mix']


def backbone(variant='dense', enabled=True, rho=0.25):
    return TransReID(
        img_size=(32, 16), patch_size=16, stride_size=16, embed_dim=8,
        depth=4, num_heads=2, mlp_ratio=2, qkv_bias=True, drop_path_rate=0,
        token_trajectory=enabled, token_trajectory_variant=variant,
        token_trajectory_ema_decay=rho,
        token_trajectory_hidden_dim=4, token_trajectory_rank=4)


@pytest.mark.parametrize('variant', ['dense'] + VARIANTS)
def test_zero_start_preserves_backbone_parameters_output_and_rng(variant):
    torch.manual_seed(1234)
    base = backbone(enabled=False).eval()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(1234)
    candidate = backbone(variant).eval()
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for name, tensor in base.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    x = torch.randn(2, 3, 32, 16)
    with torch.no_grad():
        assert torch.equal(base(x), candidate(x))


@pytest.mark.parametrize('variant', VARIANTS)
def test_learning_gate_off_and_checkpoint_roundtrip(variant):
    torch.manual_seed(7)
    model = backbone(variant).train()
    x = torch.randn(3, 3, 32, 16)
    target = torch.randn(3, 8)
    with torch.no_grad():
        initial_off = model(x, trajectory_gate=torch.zeros(3))
    optimizer = torch.optim.Adam(model.token_trajectory.parameters(), lr=0.01)
    reached = set()
    # Zero gain/output projections intentionally delay some upstream gradients.
    for _ in range(4):
        model.zero_grad()
        loss = (model(x) - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in model.token_trajectory.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
            if p.grad.abs().sum() > 0:
                reached.add(name)
        optimizer.step()
    assert reached == set(dict(model.token_trajectory.named_parameters()))
    model.eval()
    with torch.no_grad():
        assert torch.equal(initial_off, model(x, trajectory_gate=torch.zeros(3)))
        prediction = model(x)
        assert not torch.allclose(initial_off, prediction)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = backbone(variant).eval()
        restored.load_state_dict(torch.load(buffer), strict=True)
        assert torch.equal(prediction, restored(x))


def test_adaptive_starts_as_t0_and_can_distinguish_cls_from_patches():
    module = TrajectoryVariant(4, 3, 8, 'adaptive')
    original = DenseCrossLayerTokenTrajectory(4, 3, 8)
    v, old = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    with torch.no_grad():
        module.gain.normal_()
        original.gain.copy_(module.gain)
        assert torch.equal(module(2, v, old), original(2, v, old))
        module.beta_coeff[0, 0, 0] = 1
        beta = module.adaptive_beta(2, v, old)
        assert (beta[:, 0] > 1).all()
        assert (beta[:, 1:] == 1).all()


def test_ema_observes_corrected_block_input_and_resets_each_forward():
    model = backbone('ema').eval()
    with torch.no_grad():
        model.token_trajectory.gain.normal_(std=0.1)
    raw, observed = [], []
    handles = [block.register_forward_hook(
        lambda block, inputs, output: raw.append((output - inputs[0]).detach()))
        for block in model.blocks]
    handles.append(model.token_trajectory.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[1].detach())))
    x = torch.randn(2, 3, 32, 16)
    with torch.no_grad():
        first = model(x)
        moment = raw[0]
        assert torch.equal(observed[0], moment)
        for layer in range(1, len(observed)):
            moment = 0.25 * moment + 0.75 * raw[layer]
            assert torch.equal(observed[layer], moment)
        second = model(x)
        assert torch.equal(first, second)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize('variant,expected', [
    ('adaptive', 1089852), ('split', 2080512), ('ema', 1089792),
    ('token_gate', 1360491), ('channel_mix', 1360128)])
def test_actual_parameter_budget(variant, expected):
    module = TrajectoryVariant(12, 129, 768, variant)
    assert module.trajectory_parameters == expected


@pytest.mark.parametrize('source,target,source_rho,target_rho', [
    ('dense', 'ema', 0.25, 0.25), ('ema', 'dense', 0.25, 0.25),
    ('adaptive', 'split', 0.25, 0.25), ('ema', 'ema', 0.25, 0.4)])
def test_test_entry_loader_rejects_wrong_method_before_copying(
        tmp_path, source, target, source_rho, target_rho):
    source_model = torch.nn.Module()
    source_model.base = backbone(source, rho=source_rho)
    target_model = torch.nn.Module()
    target_model.base = backbone(target, rho=target_rho)
    before = {k: v.clone() for k, v in target_model.state_dict().items()}
    path = tmp_path / 'model.pth'
    torch.save(source_model.state_dict(), path)
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        build_transformer.load_param(target_model, path)
    assert all(torch.equal(v, target_model.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize('variant', VARIANTS)
def test_test_entry_loader_accepts_same_method(tmp_path, variant):
    source = torch.nn.Module()
    source.base = backbone(variant)
    target = torch.nn.Module()
    target.base = backbone(variant)
    with torch.no_grad():
        source.base.token_trajectory.gain.normal_()
    path = tmp_path / 'model.pth'
    torch.save(source.state_dict(), path)
    build_transformer.load_param(target, path)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in source.state_dict().items())


def test_config_selectors_reject_ignored_knobs_and_mixed_actuators():
    pytest.importorskip('yacs')
    from config import cfg
    from model.backbones.token_trajectory_variants import validate_trajectory_config
    for key, value in [('TOKEN_TRAJECTORY', False), ('TOKEN_TRAJECTORY_ACCEL_MIX', 0.5),
                       ('TOKEN_TRAJECTORY_RANK', 32), ('TOKEN_TRAJECTORY_HIDDEN_DIM', 32),
                       ('TOKEN_TRAJECTORY_EMA_DECAY', 0.5), ('VPR', True),
                       ('MOD_DELTA', True), ('TEXT_ALIGN', True)]:
        candidate = cfg.clone()
        candidate.MODEL.TOKEN_TRAJECTORY = True
        candidate.MODEL.TOKEN_TRAJECTORY_VARIANT = 'adaptive'
        setattr(candidate.MODEL, key, value)
        with pytest.raises(ValueError):
            validate_trajectory_config(candidate)
