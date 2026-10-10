"""M2-4 epoch-boundary restore and original C0 student export."""
import copy
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from yacs.config import CfgNode
from utils.m4_artifacts import VERSION, BASE_COMMIT, file_hash

def attach_metadata(model, cfg, bank):
    model.m4_metadata = {'variant': 'clip_relation_distill', 'implementation': VERSION,
        'base_commit': BASE_COMMIT, 'config': cfg.dump(), 'teacher': bank['metadata'],
        'teacher_feature_images': bank['teacher_feature_images'], 'preparation_optimizer_updates': 0,
        'relation_sha256': file_hash(cfg.M2.RELATION_BANK), 'centers_sha256': bank['centers_sha256'],
        'feature_sha256': bank['feature_sha256'],
        'student': {'trajectory_variant': cfg.MODEL.TOKEN_TRAJECTORY_VARIANT,
                    'acceleration_mix': cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX,
                    'gain_shape': list(model.base.token_trajectory.gain.shape)}}

def save_training(output, name, epoch, model, optimizer, scheduler, scaler, iterations, updates, amp_skips):
    meta = copy.deepcopy(model.m4_metadata)
    meta['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
    meta.update(epoch=epoch, actual_iterations=iterations, optimizer_updates=updates, amp_skips=amp_skips)
    visual = model.state_dict()  # No teacher/cache tensor is registered in the M2-4 student.
    torch.save({'state_dict': visual, 'metadata': meta,
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                'numpy': np.random.get_state(), 'python': random.getstate()}},
        os.path.join(output, '{}_{}.pth'.format(name, epoch)))
    torch.save({'state_dict': visual, 'metadata': meta}, os.path.join(output, '{}_{}_student.pth'.format(name, epoch)))

def restore_training(path, model, optimizer, scheduler, scaler):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    meta = obj['metadata']
    for key in ('variant', 'implementation', 'student', 'relation_sha256', 'centers_sha256', 'feature_sha256'):
        if meta[key] != model.m4_metadata[key]:
            raise ValueError('M2-4 resume manifest differs: ' + key)
    old, new = CfgNode.load_cfg(meta['config']), CfgNode.load_cfg(model.m4_metadata['config'])
    for config in (old, new):
        config.OUTPUT_DIR = ''
        config.M2.RESUME = ''
    if old != new:
        raise ValueError('M2-4 resume training settings differ')
    model.load_state_dict(obj['state_dict'], strict=True)
    optimizer.load_state_dict(obj['optimizer'])
    scheduler.load_state_dict(obj['scheduler'])
    scaler.load_state_dict(obj['scaler'])
    torch.set_rng_state(obj['rng']['torch'])
    torch.cuda.set_rng_state_all(obj['rng']['cuda'])
    np.random.set_state(obj['rng']['numpy'])
    random.setstate(obj['rng']['python'])
    return meta['epoch'] + 1, meta['actual_iterations'], meta['optimizer_updates'], meta['amp_skips']
