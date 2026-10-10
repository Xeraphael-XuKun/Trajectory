"""M2-1 train-only source and condition prototype correspondence."""
import hashlib
import re
from pathlib import Path
import torch

VERSION = 'm2-1-audit-v2'
BASE_COMMIT = 'a8349abeb5fb62b5f7d233e33202d1ef1d113b42'
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

def train_manifest(root, subdir='WHU-MARS'):
    rows = []
    for mod, name in enumerate(('RGB', 'IR', 'Thermal')):
        for path in sorted((Path(root) / (subdir or 'WHU-MARS') / 'train' / name).glob('*.jpg')):
            pid, camera = map(int, re.search(r'(\d+)_c(\d+)', path.name).groups())
            rows.append((str(path), pid, camera - 1, mod))
    if not rows:
        raise ValueError('no train images found')
    mapping = {pid: label for label, pid in enumerate(sorted({r[1] for r in rows}))}
    return [(p, mapping[y], c, m) for p, y, c, m in rows], mapping

def describe(rows, mapping, clip):
    mask = torch.zeros(len(mapping), 3, 2, dtype=torch.bool)
    for _, y, c, m in rows:
        mask[y, m, int(c in (5, 6))] = True
    return {'split': 'train', 'pid_map': mapping,
            'filenames': [['RGB', 'IR', 'Thermal'][r[3]] + '/' + Path(r[0]).name for r in rows],
            'modality_names': ['RGB', 'IR', 'Thermal'], 'condition_mask': mask,
            'clip_sha256': file_hash(clip), 'clip_path': str(clip), 'preprocess': PREPROCESS.copy()}

def validate_source(meta, root, subdir, clip):
    rows, mapping = train_manifest(root, subdir)
    expected = describe(rows, mapping, clip)
    for key in ('split', 'pid_map', 'filenames', 'modality_names', 'clip_sha256', 'preprocess'):
        if meta.get(key) != expected[key]:
            raise ValueError('M2-1 artifact source differs: ' + key)
    if not torch.equal(meta['condition_mask'].cpu(), expected['condition_mask']):
        raise ValueError('M2-1 condition existence differs from current train')
    return rows, mapping

def validate_cache(cache, root, subdir, clip):
    if 'metadata' not in cache:
        raise ValueError('old cache lacks train provenance; rebuild it')
    rows, mapping = validate_source(cache['metadata'], root, subdir, clip)
    for key, values in [('pid', [r[1] for r in rows]), ('cams', [r[2] for r in rows]),
                        ('modality', [r[3] for r in rows]), ('platform', [int(r[2] in (5, 6)) for r in rows])]:
        if not torch.equal(cache[key].cpu(), torch.tensor(values)):
            raise ValueError('M2-1 cache row correspondence differs: ' + key)
    if cache['feature'].shape != (len(rows), 512) or not torch.isfinite(cache['feature']).all():
        raise ValueError('invalid teacher features')
    return mapping

def validate_bank(bank, cfg, num_classes):
    if bank.get('variant') != 'condition_text_bridge' or bank.get('implementation') != VERSION:
        raise ValueError('old/wrong M2-1 bank; rerun stage A')
    _, mapping = validate_source(bank['metadata'], cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR, cfg.M2.CLIP_PATH)
    if len(mapping) != num_classes or bank['stage_config'] != dict(cfg.PROMPT_STAGE):
        raise ValueError('M2-1 bank identity count or prompt recipe differs')
    if bank['updates'] != cfg.PROMPT_STAGE.MAX_UPDATES or bank['seed'] != cfg.SOLVER.SEED:
        raise ValueError('M2-1 bank actual budget/seed differs')
    from model.m2.condition_prompt import prompt_signature
    if bank['signature'] != prompt_signature():
        raise ValueError('M2-1 template/tokenizer differs')
    expected = bank['metadata']['condition_mask'].nonzero(as_tuple=False)
    actual = torch.stack([bank['pid'], bank['modality'], bank['platform']], dim=1).cpu()
    if not torch.equal(actual, expected):
        raise ValueError('M2-1 bank must contain exactly the observed train conditions')
    if bank['text'].shape != (len(expected), 512) or not torch.isfinite(bank['text']).all():
        raise ValueError('invalid condition text bank')
    torch.testing.assert_close(bank['text'].float().norm(dim=1), torch.ones(len(expected)), rtol=1e-4, atol=1e-5)
