"""M2-2 epoch-boundary training restore and original-C0 deployment export."""
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
    meta['code_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
    meta.update(epoch=epoch, actual_iterations=iterations, optimizer_updates=updates, amp_skips=amp_skips)
    torch.save({'state_dict': model.state_dict(), 'metadata': meta,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                        'numpy': np.random.get_state(), 'python': random.getstate()}},
               os.path.join(output, '{}_{}.pth'.format(name, epoch)))
    # Eval never needs the inversion/text/cache branch.
    state = {k: v for k, v in model.state_dict().items() if k.startswith(('base.', 'bottleneck.', 'classifier.'))}
    torch.save({'state_dict': state, 'metadata': meta},
               os.path.join(output, '{}_{}_student.pth'.format(name, epoch)))

def restore_training(path, model, optimizer, scheduler, scaler):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    meta = obj['metadata']
    for key in ('variant', 'implementation', 'summary_width', 'pseudo_tokens', 'slots', 'eot', 'rgb_center_sha256'):
        if meta[key] != model.m2_metadata[key]:
            raise ValueError('M2-2 resume metadata mismatch: {}'.format(key))
    # OUTPUT_DIR / RESUME are operational; all training settings must match.
    from yacs.config import CfgNode
    old, new = CfgNode.load_cfg(meta['config']), CfgNode.load_cfg(model.m2_metadata['config'])
    for config in (old, new):
        config.OUTPUT_DIR = ''
        config.M2.RESUME = ''
    if old != new:
        raise ValueError('M2-2 resume training config mismatch')
    model.load_state_dict(obj['state_dict'], strict=True)
    optimizer.load_state_dict(obj['optimizer'])
    scheduler.load_state_dict(obj['scheduler'])
    scaler.load_state_dict(obj['scaler'])
    torch.set_rng_state(obj['rng']['torch'])
    torch.cuda.set_rng_state_all(obj['rng']['cuda'])
    np.random.set_state(obj['rng']['numpy'])
    random.setstate(obj['rng']['python'])
    return meta['epoch'] + 1, meta['actual_iterations'], meta['optimizer_updates'], meta['amp_skips']
