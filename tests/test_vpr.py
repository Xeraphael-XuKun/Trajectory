"""Unit tests for VPR invariants and loss scaling.

Run in the training environment (PyTorch is required):
    pytest -q tests/test_vpr.py
"""

import torch

from loss.vpr_losses import (vpr_cross_spectral_displacement_loss,
                             vpr_view_text_loss)
from model.backbones.vpr import (ViewAwarePositionalResidual,
                                 low_frequency_dct_basis)


def test_dct_basis_has_no_dc_component():
    basis = low_frequency_dct_basis(8, 4, rank=10)
    assert basis.shape == (32, 10)
    assert torch.allclose(basis.mean(0), torch.zeros(10), atol=1e-6)
    assert torch.allclose(basis.norm(dim=0), torch.ones(10), atol=1e-6)


def test_vpr_is_exact_zero_at_initialisation_and_patch_only():
    module = ViewAwarePositionalResidual(
        depth=3, embed_dim=8, grid_h=8, grid_w=4, rank=6)
    cls = torch.randn(2, 8)
    field = module(0, cls, 8, 4)
    assert field.shape == (2, 32, 8)  # no CLS row
    assert torch.count_nonzero(field) == 0
    assert module.field_parameters == 3 * 6 * 8


def test_runtime_grid_is_continuous_and_zero_mean():
    module = ViewAwarePositionalResidual(
        depth=2, embed_dim=4, grid_h=8, grid_w=4, rank=5)
    with torch.no_grad():
        module.coeff[0].normal_()
    field = module(0, torch.randn(3, 4), 10, 5)
    assert field.shape == (3, 50, 4)
    assert torch.allclose(field.mean(1), torch.zeros(3, 4), atol=1e-5)


def test_view_text_has_finite_nonzero_step_zero_gradient():
    before = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
    after = before.clone().requires_grad_(True)
    aux = {
        'feat_before': before,
        'feat_after': after,
        'proj': torch.eye(2, requires_grad=True),
        'text': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        'text_rows': torch.tensor([True, True]),
        'is_aerial': torch.tensor([True, False]),
    }
    loss, stats = vpr_view_text_loss(aux)
    assert loss.item() > 0.1  # vector-energy loss, not /512 coordinate MSE
    loss.backward()
    assert torch.isfinite(after.grad).all()
    assert after.grad[0].abs().sum() > 0
    # Projection is a fixed measuring coordinate for this loss.
    assert aux['proj'].grad is None
    assert stats['n_aerial'] == 1 and stats['n_ground'] == 1


def _csd_aux(after):
    before = torch.tensor([[1.0, 0.0]]).repeat(3, 1)
    return {
        'feat_before': before,
        'feat_after': after,
        'proj': torch.eye(2),
        'is_aerial': torch.tensor([True, True, True]),
        'n_modality': 3,
        'per_modality': 1,
    }


def test_cross_spectral_displacement_matches_same_capture_rgb():
    same = torch.tensor([[0.0, 1.0]]).repeat(3, 1).requires_grad_(True)
    loss_same, _ = vpr_cross_spectral_displacement_loss(_csd_aux(same))
    assert torch.allclose(loss_same, torch.zeros_like(loss_same), atol=1e-7)

    different = torch.tensor([[0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
                             requires_grad=True)
    loss_diff, stats = vpr_cross_spectral_displacement_loss(_csd_aux(different))
    assert loss_diff.item() > 0.1
    loss_diff.backward()
    assert torch.isfinite(different.grad).all()
    # RGB teacher is stop-gradient under CSD; the other spectra are students.
    assert different.grad[0].abs().sum() == 0
    assert stats['n_valid'] == 1
