"""Read-only bridge to the audited old M5 resources; legacy loss stays disabled."""
import hashlib
import json
import torch
from tools.m2.artifacts import train_source, validate_text_bank, file_digest

METHOD = 'camera-view-paired-id-neutral-v1'
TEACHER_COMMIT = 'db8a5c65ba0b7b5f0c62d00d2adb2f5e3bfe1177'


def legacy_config(c):
    old = c.clone(); old.defrost()
    old.ICLR_EXP.ENABLED = False
    old.M2.ENABLED = True
    old.M2.VARIANT = 'conditional_teacher_distill'
    old.M2.TRAIN_STAGE = 'student'
    old.M2.CLIP_PATH = c.ICLR_EXP.ASSETS.ORIGINAL_CLIP
    old.M2.TEXT_BANK = c.ICLR_EXP.ASSETS.TEXT_BANK
    old.M2.PROMPT_WEIGHT = c.ICLR_EXP.ASSETS.PROMPT_CKPT
    old.M2.TEACHER_WEIGHT = c.ICLR_EXP.ASSETS.TEACHER_CKPT
    return old


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def fingerprint(c, teacher_meta, source):
    return dict(teacher_sha256=file_digest(c.ICLR_EXP.ASSETS.TEACHER_CKPT),
        text_bank_sha256=file_digest(c.ICLR_EXP.ASSETS.TEXT_BANK),
        prompt_sha256=file_digest(c.ICLR_EXP.ASSETS.PROMPT_CKPT),
        original_clip_sha256=source['clip_sha256'],
        teacher_train_commit=teacher_meta['code_commit'],
        pid_mapping=source['pid_map'], pid_mapping_sha256=json_digest(source['pid_map']),
        train_manifest_sha256=json_digest(source['filenames']),
        modality_order=list(c.DATASETS.MODALITIES), platform_order=['Ground','Aerial'],
        preprocess_spec=source['preprocess'])


def load_resources(c, device):
    from utils.m5_checkpoint import load_teacher
    old = legacy_config(c)
    bank = torch.load(old.M2.TEXT_BANK, map_location='cpu', weights_only=False)
    source = validate_text_bank(bank, old, 500)
    # Complete teacher (including classifier/BN) and frozen-original correspondence.
    teacher, obj = load_teacher(old, 500, device)
    if obj['metadata']['code_commit'] != TEACHER_COMMIT:
        raise ValueError('teacher is not the selected audited old M5 epoch60 resource')
    return teacher, bank['text_bank'].detach().float().to(device), source, obj['metadata']


def load_shift(c, source, teacher_meta, device):
    obj = torch.load(c.ICLR_EXP.ASSETS.NEUTRAL_STATS, map_location='cpu', weights_only=False)
    expected = fingerprint(c, teacher_meta, source)
    for key, value in expected.items():
        if obj[key] != value:
            raise ValueError('neutral statistics provenance differs: '+key)
    fixed = dict(schema_version=1, method_version=METHOD, complete=True,
        num_train_images=92133, num_train_ids=500, base_temperature=.07,
        kd_temperature_reference=2., class_centered=True,
        fit_input='deterministic_clean_train', inference_precision='FP32 visual; FP64 sums',
        estimator='camera_then_view_then_paired_identity_equal_mean', coefficient=1.)
    for key, value in fixed.items():
        if obj.get(key) != value:
            raise ValueError('neutral statistics definition differs: '+key)
    shift = obj['shift_fp32']
    counts = obj['group_counts']
    if shift.shape != (3,500) or not torch.isfinite(shift).all() or counts.sum()!=92133:
        raise ValueError('invalid neutral statistics shape/count/values')
    if not torch.equal(shift, obj['shift_fp64'].float()) or shift.double().mean(-1).abs().max()>1e-6:
        raise ValueError('neutral statistics precision/centering differs')
    if any(n<=0 for n in obj['num_paired_ids_by_modality']):
        raise ValueError('a modality has no paired identities: neutral statistics unavailable')
    return shift.to(device).detach(), obj


def paired_mask(stats, device):
    mask = torch.zeros(500,3,dtype=torch.bool,device=device)
    for m, ids in enumerate(stats['paired_ids_by_modality']):
        mask[ids,m] = True
    return mask
