"""M2-2 train manifest and fixed teacher provenance."""
import hashlib
import re
from pathlib import Path
import torch

PREPROCESS = {'size': [256, 128], 'interpolation': 'bilinear',
              'mean': [0.48145466, 0.4578275, 0.40821073],
              'std': [0.26862954, 0.26130258, 0.27577711],
              'ln_pre': True, 'activation': 'QuickGELU', 'use_c0': False}

def weight_id(path):
    # Required once for teacher/cache provenance, not for every train step.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def train_manifest(root, subdir='WHU-MARS', modalities=('RGB', 'IR', 'Thermal')):
    train_dir = Path(root) / (subdir or 'WHU-MARS') / 'train'
    rows = []
    pattern = re.compile(r'(\d+)_c(\d+)')
    for mod, name in enumerate(modalities):
        for path in sorted((train_dir / name).glob('*.jpg')):
            pid, cam = map(int, pattern.search(path.name).groups())
            rows.append((str(path), pid, cam - 1, mod))
    if not rows:
        raise ValueError('no train images in {}'.format(train_dir))
    pid_map = {pid: label for label, pid in enumerate(sorted({r[1] for r in rows}))}
    rows = [(path, pid_map[pid], cam, mod) for path, pid, cam, mod in rows]
    return rows, pid_map

def metadata(rows, pid_map, modalities, clip_path):
    mask = torch.zeros(len(pid_map), len(modalities), 2, dtype=torch.bool)
    for _, pid, cam, mod in rows:
        mask[pid, mod, int(cam in (5, 6))] = True
    return {'split': 'train', 'pid_map': pid_map,
            'filenames': [str(Path(modalities[r[3]]) / Path(r[0]).name) for r in rows],
            'modality_names': list(modalities), 'condition_mask': mask,
            'clip_path': str(clip_path), 'clip_sha256': weight_id(clip_path),
            'preprocess': PREPROCESS.copy()}

def validate_centers(obj, cfg, num_classes):
    if not isinstance(obj, dict) or 'metadata' not in obj:
        raise ValueError('RGB centers need provenance metadata; rebuild the cache')
    meta = obj['metadata']
    rows, pid_map = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR,
                                   cfg.DATASETS.MODALITIES)
    names = [str(Path(cfg.DATASETS.MODALITIES[r[3]]) / Path(r[0]).name) for r in rows]
    if (meta['split'] != 'train' or meta['pid_map'] != pid_map or
            meta['filenames'] != names or
            meta['modality_names'] != list(cfg.DATASETS.MODALITIES)):
        raise ValueError('RGB center train/PID/modality mapping differs from current data')
    if (meta['preprocess'] != PREPROCESS or
            meta['clip_sha256'] != weight_id(cfg.M2.CLIP_PATH)):
        raise ValueError('RGB teacher weights/preprocessing differ; rebuild the cache')
    centers = obj['centers'].float()
    if (centers.shape != (num_classes, 512) or
            not torch.equal(obj['pids'].cpu(), torch.arange(num_classes)) or
            not torch.isfinite(centers).all() or (centers.norm(dim=1) == 0).any()):
        raise ValueError('RGB centers must cover each train label once, in label order')
    if not torch.equal(meta['condition_mask'][:, 0].any(1), torch.ones(num_classes, dtype=torch.bool)):
        raise ValueError('a train identity has no RGB observation')
    return centers, meta
