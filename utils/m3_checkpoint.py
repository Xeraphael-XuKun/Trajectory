"""M2-3 complete epoch restore and controller-preserving deployment."""
import copy
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import torch

def save_training(output, name, epoch, model, optimizer, scheduler, scaler,
                  iterations, updates, amp_skips):
    meta = copy.deepcopy(model.m2_metadata)
    meta['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                        cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
    meta.update(epoch=epoch, actual_iterations=iterations, optimizer_updates=updates, amp_skips=amp_skips)
    torch.save({'state_dict': model.state_dict(), 'metadata': meta,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                        'numpy': np.random.get_state(), 'python': random.getstate()}},
               os.path.join(output, '{}_{}.pth'.format(name, epoch)))
    # Includes final C_l and all controller projections; excludes training ID/text.
    visual = {key: value for key, value in model.state_dict().items()
              if key.startswith(('base.', 'bottleneck.', 'classifier.'))}
    torch.save({'state_dict': visual, 'metadata': meta},
               os.path.join(output, '{}_{}_student.pth'.format(name, epoch)))

def restore_training(path, model, optimizer, scheduler, scaler, allow_preflight=False):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    meta = obj['metadata']
    if meta.get('r2_preflight_only') and not allow_preflight:
        raise ValueError('R2 preflight checkpoint cannot resume formal training')
    if meta.get('r2') != model.m2_metadata.get('r2'):
        raise ValueError('R2 resume experiment/parent/weight decay differs; old M3 cannot resume E3')
    for key in ('variant', 'implementation', 'rho', 'control_width', 'context_length',
                'acceleration_mix', 'signature', 'prompt_sha256', 'prompt_updates'):
        if meta[key] != model.m2_metadata[key]:
            raise ValueError('M2-3 resume manifest differs: {}'.format(key))
    from yacs.config import CfgNode
    old = CfgNode.load_cfg(meta['config'])
    new = CfgNode.load_cfg(model.m2_metadata['config'])
    for config in (old, new):
        config.OUTPUT_DIR = ''
        config.M2.RESUME = ''
        if 'R2' in config and not config.R2.ENABLED:
            del config['R2']
    if old != new:
        raise ValueError('M2-3 resume training settings differ')
    model.load_state_dict(obj['state_dict'], strict=True)
    optimizer.load_state_dict(obj['optimizer'])
    scheduler.load_state_dict(obj['scheduler'])
    scaler.load_state_dict(obj['scaler'])
    torch.set_rng_state(obj['rng']['torch'])
    torch.cuda.set_rng_state_all(obj['rng']['cuda'])
    np.random.set_state(obj['rng']['numpy'])
    random.setstate(obj['rng']['python'])
    for key in ('r2_initial_state_sha256', 'r2_stage_seconds'):
        if key in meta:
            model.m2_metadata[key] = meta[key]
    return meta['epoch'] + 1, meta['actual_iterations'], meta['optimizer_updates'], meta['amp_skips']
