"""M2-5 stage manifests, strict teacher load, epoch-boundary restore/export."""
import copy
import os
import torch
from yacs.config import CfgNode
from tools.m2.artifacts import VERSION, BASE_COMMIT, code_commit, file_digest, rng_state, restore_rng, validate_text_bank

def compare_config(old, new):
    a, b = CfgNode.load_cfg(old), CfgNode.load_cfg(new)
    for c in (a, b):
        c.OUTPUT_DIR = ''
        c.TEACHER_STAGE.OUTPUT_DIR = ''
        c.M2.RESUME = ''
        c.M2.PROMPT_RESUME = ''
    if a != b:
        raise ValueError('M2-5 resume training settings differ')

def architecture(c):
    return {'depth': 12, 'dim': 768, 'targets': list(c.TEACHER_STAGE.LORA_TARGETS),
        'rank': c.TEACHER_STAGE.LORA_RANK, 'alpha': c.TEACHER_STAGE.LORA_ALPHA,
        'modalities': c.TEACHER_STAGE.MODALITY_BANKS, 'platforms': c.TEACHER_STAGE.PLATFORM_BANKS,
        'use_c0': False, 'frozen_original': True, 'size': list(c.INPUT.SIZE_TRAIN),
        'stride': list(c.MODEL.STRIDE_SIZE), 'drop_path': c.MODEL.DROP_PATH,
        'text_temperature': c.TEACHER_STAGE.TEMPERATURE, 'text_weight': c.TEACHER_STAGE.TEXT_ID_WEIGHT}

def attach_metadata(model, c, bank, teacher=None):
    model.m5_metadata = {'variant': c.M2.VARIANT, 'implementation': VERSION,
        'base_commit': BASE_COMMIT, 'stage': c.M2.TRAIN_STAGE, 'config': c.dump(),
        'source': bank['metadata']['source'], 'text_metadata': bank['metadata'],
        'text_sha256': file_digest(c.M2.TEXT_BANK), 'teacher_architecture': architecture(c),
        'student': {'trajectory_variant': c.MODEL.TOKEN_TRAJECTORY_VARIANT,
            'acceleration_mix': c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX,
            'gain_shape': list(model.base.token_trajectory.gain.shape)} if c.M2.TRAIN_STAGE == 'student' else None}
    if teacher is not None:
        model.m5_metadata['teacher_sha256'] = file_digest(c.M2.TEACHER_WEIGHT)
        model.m5_metadata['teacher_metadata'] = teacher['metadata']

def load_teacher(c, num_classes, device):
    from model.m2_teacher import ConditionalVisualTeacher
    obj = torch.load(c.M2.TEACHER_WEIGHT, map_location='cpu', weights_only=False)
    meta = obj['metadata']
    if (meta['implementation'] != VERSION or meta['variant'] != c.M2.VARIANT or meta['stage'] != 'teacher' or
        meta['epoch'] != c.TEACHER_STAGE.MAX_EPOCHS or meta['teacher_architecture'] != architecture(c) or
        meta['text_sha256'] != file_digest(c.M2.TEXT_BANK)):
        raise ValueError('M2-5 teacher stage/budget/architecture/shared text differs')
    bank = torch.load(c.M2.TEXT_BANK, map_location='cpu', weights_only=False)
    validate_text_bank(bank, c, num_classes)
    # Teacher's source must agree with the bank/current train, not merely claim train.
    from tools.m2.artifacts import validate_source
    validate_source(meta['source'], c)
    with torch.random.fork_rng(devices=[]):
        model = ConditionalVisualTeacher(c, num_classes, load_original=True)
    for key, value in model.base.state_dict().items():
        # fc is the unused random placeholder, not a tensor loaded from CLIP.
        if not key.startswith('fc.') and not torch.equal(value.cpu(), obj['state_dict']['base.' + key].cpu()):
            raise ValueError('M2-5 teacher frozen original visual changed: ' + key)
    model.load_state_dict(obj['state_dict'], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, obj

def save_training(output, name, epoch, model, optimizer, scheduler, scaler, iterations, updates, amp_skips, seconds):
    meta = copy.deepcopy(model.m5_metadata)
    meta.update(code_commit=code_commit(), epoch=epoch, actual_iterations=iterations,
        optimizer_updates=updates, amp_skips=amp_skips, stage_seconds=seconds)
    state = model.state_dict()
    stem = 'conditional_teacher_epoch%d' % epoch if meta['stage'] == 'teacher' else '%s_%d' % (name, epoch)
    payload = {'state_dict': state, 'metadata': meta, 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state()}
    torch.save(payload, os.path.join(output, stem + '.pth'))
    if meta['stage'] == 'student':
        torch.save({'state_dict': state, 'metadata': meta}, os.path.join(output, stem + '_student.pth'))
    return payload

def restore_training(path, model, optimizer, scheduler, scaler):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    meta = obj['metadata']
    for key in ('variant', 'implementation', 'stage', 'text_sha256', 'teacher_architecture', 'student', 'teacher_sha256'):
        if meta.get(key) != model.m5_metadata.get(key):
            raise ValueError('M2-5 resume manifest differs: ' + key)
    compare_config(meta['config'], model.m5_metadata['config'])
    model.load_state_dict(obj['state_dict'], strict=True)
    optimizer.load_state_dict(obj['optimizer'])
    scheduler.load_state_dict(obj['scheduler'])
    scaler.load_state_dict(obj['scaler'])
    restore_rng(obj['rng'])
    return meta['epoch'] + 1, meta['actual_iterations'], meta['optimizer_updates'], meta['amp_skips'], meta['stage_seconds']
