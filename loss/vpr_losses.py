"""The two auxiliary objectives used by VPR.

Both operate on the *displacement* caused by switching VPR on for the same
image.  ``feat_before`` is stop-gradient and the CLIP projection/text anchors
are fixed, preventing the reference coordinate system from chasing the
student.  Their outer weights live in the config and are 5.0 in the released
experiment, matching the preceding VTC implementation.
"""

import torch
import torch.nn.functional as F


def _zero_with_graph(aux):
    return aux['feat_after'].sum() * 0.0


def _joint_features(feat, projection):
    """Map 768-D visual features into frozen CLIP joint space."""
    if projection is None:
        raise ValueError('VPR losses require the CLIP visual projection')
    # The map defines the measuring coordinate system; its values are not an
    # optimisation target of these auxiliary losses.
    z = feat.float() @ projection.detach().float()
    return F.normalize(z, dim=-1)


def vpr_view_text_loss(aux, orth_weight=1.0, ground_weight=1.0):
    """Fixed aerial->ground direction regression plus a ground drift guard.

    The main direction term is squared vector error, not cosine at zero.  A
    freshly initialised VPR produces zero displacement; cosine's derivative is
    ill-conditioned there, whereas this regression has a finite, non-zero
    gradient from the first step.
    """
    select = aux['text_rows'].bool()
    if not bool(select.any()):
        return _zero_with_graph(aux), {'n_aerial': 0, 'n_ground': 0}

    after = _joint_features(aux['feat_after'][select], aux['proj'])
    before = _joint_features(aux['feat_before'][select].detach(), aux['proj'])
    displacement = after - before
    is_aerial = aux['is_aerial'][select].bool()

    text = F.normalize(aux['text'].detach().float(), dim=-1)
    if text.shape[0] != 2:
        raise ValueError('VPR view-text expects [aerial, ground] anchors, got {}'
                         .format(tuple(text.shape)))
    text_delta = text[1] - text[0]
    target_norm = text_delta.norm()
    if float(target_norm) < 1e-6:
        raise RuntimeError('aerial and ground text anchors collapsed')
    # Unit length removes the arbitrary magnitude of CLIP's two sentence
    # embeddings.  At zero VPR displacement the raw direction loss is exactly
    # 1.0, so the configured x5 contribution is genuinely order five rather
    # than a nominal weight multiplying a tiny target vector.
    target = text_delta / target_norm
    target_hat = target

    zero = displacement.sum() * 0.0
    if bool(is_aerial.any()):
        aerial_d = displacement[is_aerial]
        # Sum over the 512 coordinates, mean over images: O(1), not a tiny
        # per-coordinate MSE divided by 512.
        direction = (aerial_d - target).square().sum(dim=-1).mean()
        parallel = (aerial_d * target_hat).sum(dim=-1, keepdim=True) * target_hat
        orthogonal = (aerial_d - parallel).square().sum(dim=-1).mean()
    else:
        direction = orthogonal = zero

    is_ground = ~is_aerial
    ground_keep = (displacement[is_ground].square().sum(dim=-1).mean()
                   if bool(is_ground.any()) else zero)
    total = direction + float(orth_weight) * orthogonal + \
        float(ground_weight) * ground_keep

    with torch.no_grad():
        aerial_d = displacement[is_aerial]
        direction_cos = (F.cosine_similarity(
            aerial_d, target.unsqueeze(0), dim=-1).mean().item()
            if aerial_d.numel() else float('nan'))
        stats = {
            'direction': float(direction.detach()),
            'orthogonal': float(orthogonal.detach()),
            'ground_keep': float(ground_keep.detach()),
            'direction_cos': direction_cos,
            'displacement_norm': (float(aerial_d.norm(dim=-1).mean())
                                  if aerial_d.numel() else float('nan')),
            'target_norm': float(target_norm),
            'n_aerial': int(is_aerial.sum()),
            'n_ground': int(is_ground.sum()),
        }
    return total, stats


def vpr_cross_spectral_displacement_loss(aux, reference_index=0):
    """Make synchronized spectra share the RGB VPR displacement.

    Rows must be modality-major ``[M * B, D]``.  The reference displacement is
    stop-gradient; every other spectrum regresses to the same-capture RGB
    correction.  Non-reference terms are summed, matching the preceding VTC's
    summed-cell convention instead of shrinking the loss as modalities grow.
    """
    n_modality = int(aux['n_modality'])
    per_modality = int(aux['per_modality'])
    expected = n_modality * per_modality
    if aux['feat_after'].shape[0] != expected:
        raise ValueError('CSD expects all modality-major rows: {} != {} x {}'
                         .format(aux['feat_after'].shape[0], n_modality,
                                 per_modality))
    if n_modality < 2:
        raise ValueError('CSD needs at least two spectra')
    if reference_index < 0 or reference_index >= n_modality:
        raise ValueError('CSD reference index out of range')

    after = _joint_features(aux['feat_after'], aux['proj'])
    before = _joint_features(aux['feat_before'].detach(), aux['proj'])
    displacement = (after - before).reshape(n_modality, per_modality, -1)
    aerial = aux['is_aerial'].reshape(n_modality, per_modality)
    # A synchronized tuple should agree on camera.  Requiring all spectra to be
    # aerial both validates that invariant and avoids supervising ground zeros.
    valid = aerial.all(dim=0)
    if not bool(valid.any()):
        return _zero_with_graph(aux), {
            'n_valid': 0, 'per_modality': [float('nan')] * n_modality}

    teacher = displacement[reference_index, valid].detach()
    total = displacement.sum() * 0.0
    per = []
    for modality in range(n_modality):
        if modality == reference_index:
            per.append(0.0)
            continue
        term = (displacement[modality, valid] - teacher).square().sum(
            dim=-1).mean()
        total = total + term
        per.append(float(term.detach()))
    return total, {'n_valid': int(valid.sum()), 'per_modality': per}
