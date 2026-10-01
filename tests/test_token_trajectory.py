import torch

from model.backbones.token_trajectory import DenseCrossLayerTokenTrajectory
from model.backbones.vit_pytorch import TransReID


def test_default_parameter_count_matches_declared_dense_module():
    module = DenseCrossLayerTokenTrajectory(12, 129, 768)
    assert module.trajectory_parameters == 11 * 129 * 768


def test_zero_initialisation_and_gate_off_are_exact():
    module = DenseCrossLayerTokenTrajectory(3, 5, 4)
    velocity = torch.randn(2, 5, 4)
    correction = module(1, velocity)
    assert correction.shape == velocity.shape
    assert torch.count_nonzero(correction) == 0

    with torch.no_grad():
        module.gain[0].normal_()
    off = module(1, velocity, gate=torch.zeros(2))
    assert torch.count_nonzero(off) == 0


def test_velocity_and_acceleration_drive_dense_gain_gradient():
    module = DenseCrossLayerTokenTrajectory(
        3, 5, 4, acceleration_mix=1.0)
    current = torch.randn(2, 5, 4)
    older = torch.randn(2, 5, 4)
    correction = module(2, current, older)
    loss = (correction - torch.ones_like(correction)).square().mean()
    loss.backward()
    assert module.gain.grad is not None
    assert torch.isfinite(module.gain.grad).all()
    assert module.gain.grad[1].abs().sum() > 0


def test_acceleration_changes_direction_after_gain_is_enabled():
    module = DenseCrossLayerTokenTrajectory(
        3, 5, 4, acceleration_mix=1.0)
    with torch.no_grad():
        module.gain.fill_(1.0)
    current = torch.randn(2, 5, 4)
    older = torch.randn(2, 5, 4)
    velocity_only = module(2, current, current)
    accelerated = module(2, current, older)
    assert not torch.allclose(velocity_only, accelerated)


def test_full_backbone_has_no_pld_and_gate_zero_is_exact():
    model = TransReID(
        img_size=(32, 16), patch_size=16, stride_size=16,
        embed_dim=8, depth=3, num_heads=2, mlp_ratio=2,
        qkv_bias=True, drop_path_rate=0.0,
        token_trajectory=True, token_trajectory_accel_mix=1.0)
    model.eval()
    assert not hasattr(model, 'pos_delta')
    image = torch.randn(2, 3, 32, 16)
    off_gate = torch.zeros(2)
    with torch.no_grad():
        before = model(image, trajectory_gate=off_gate)
        model.token_trajectory.gain.normal_()
        after_off = model(image, trajectory_gate=off_gate)
        after_on = model(image)
    assert torch.equal(before, after_off)
    assert not torch.allclose(after_off, after_on)
