"""M2-3 train-only teacher/prompt provenance and label correspondence."""
import hashlib
import re
from pathlib import Path
import torch

PREPROCESS = {'size': [256, 128], 'interpolation': 'bilinear',
              'mean': [0.48145466, 0.4578275, 0.40821073],
              'std': [0.26862954, 0.26130258, 0.27577711],
              'ln_pre': True, 'activation': 'QuickGELU', 'use_c0': False}

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def train_manifest(root, subdir='WHU-MARS', modalities=('RGB', 'IR', 'Thermal')):
    pattern = re.compile(r'(\d+)_c(\d+)')
    rows = []
    train_dir = Path(root) / (subdir or 'WHU-MARS') / 'train'
    for mod, name in enumerate(modalities):
        for path in sorted((train_dir / name).glob('*.jpg')):
            pid, camera = map(int, pattern.search(path.name).groups())
            rows.append((str(path), pid, camera - 1, mod))
    if not rows:
        raise ValueError('no train images in {}'.format(train_dir))
    pid_map = {pid: label for label, pid in enumerate(sorted({r[1] for r in rows}))}
    return [(path, pid_map[pid], cam, mod) for path, pid, cam, mod in rows], pid_map

def describe(rows, pid_map, clip_path, modalities=('RGB', 'IR', 'Thermal')):
    mask = torch.zeros(len(pid_map), len(modalities), 2, dtype=torch.bool)
    for _, pid, cam, mod in rows:
        mask[pid, mod, int(cam in (5, 6))] = True
    return {'split': 'train', 'pid_map': pid_map,
            'filenames': [modalities[r[3]] + '/' + Path(r[0]).name for r in rows],
            'modality_names': list(modalities), 'condition_mask': mask,
            'clip_sha256': file_hash(clip_path), 'clip_path': str(clip_path),
            'preprocess': PREPROCESS.copy()}

def validate_source(meta, root, subdir, clip_path):
    rows, pid_map = train_manifest(root, subdir)
    filenames = [['RGB', 'IR', 'Thermal'][r[3]] + '/' + Path(r[0]).name for r in rows]
    mask = torch.zeros(len(pid_map), 3, 2, dtype=torch.bool)
    for _, pid, cam, mod in rows:
        mask[pid, mod, int(cam in (5, 6))] = True
    if (meta.get('split') != 'train' or meta.get('pid_map') != pid_map or
            meta.get('filenames') != filenames or
            meta.get('modality_names') != ['RGB', 'IR', 'Thermal'] or
            not torch.equal(meta['condition_mask'], mask)):
        raise ValueError('M2-3 cache/prompt train label or condition mapping differs from current data')
    if meta['preprocess'] != PREPROCESS or meta['clip_sha256'] != file_hash(clip_path):
        raise ValueError('M2-3 teacher weights/preprocessing differ; rebuild artifacts')
    return rows, pid_map

def validate_cache(cache, root, subdir, clip_path):
    if 'metadata' not in cache:
        raise ValueError('old feature cache has no provenance; rebuild it')
    rows, pid_map = validate_source(cache['metadata'], root, subdir, clip_path)
    for key, expected in [('labels', [r[1] for r in rows]), ('cams', [r[2] for r in rows]),
                          ('modality', [r[3] for r in rows]), ('platform', [int(r[2] in (5, 6)) for r in rows])]:
        if not torch.equal(cache[key].cpu(), torch.tensor(expected)):
            raise ValueError('feature cache {} rows do not match train manifest'.format(key))
    if cache['features'].shape != (len(rows), 512) or not torch.isfinite(cache['features']).all():
        raise ValueError('invalid feature cache shape/values')
    return pid_map

def validate_bank(bank, cfg, num_classes, signature):
    if 'metadata' not in bank or 'context_init' not in bank:
        raise ValueError('M2-3 prompt bank lacks source/deep-init metadata; rerun stage A')
    if bank.get('variant') != 'deep_text_c0_control' or bank.get('implementation') != 'm2-3-audit-v2':
        raise ValueError('M2-3 prompt artifact method/version mismatch')
    _, pid_map = validate_source(bank['metadata'], cfg.DATASETS.ROOT_DIR,
                                 cfg.DATASETS.SUBDIR, cfg.M2.CLIP_PATH)
    if len(pid_map) != num_classes or bank['signature'] != signature:
        raise ValueError('M2-3 prompt identity count/template/tokenizer mismatch')
    if bank['updates'] != cfg.PROMPT_STAGE.MAX_UPDATES or bank['stage_config'] != dict(cfg.PROMPT_STAGE):
        raise ValueError('prompt budget/settings do not match configured stage A')
    if bank['seed'] != cfg.SOLVER.SEED:
        raise ValueError('prompt-stage seed mismatch')
    for key, shape in [('id_tokens', (num_classes, 4, 512)), ('anchors', (num_classes, 512)),
                       ('context_init', (11, 4, 512))]:
        if tuple(bank[key].shape) != shape or not torch.isfinite(bank[key]).all():
            raise ValueError('invalid prompt bank {}'.format(key))
