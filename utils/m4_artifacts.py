"""M2-4 train-only RGB source and camera-balanced teacher centers."""
from collections import Counter
import hashlib
import re
from pathlib import Path
import torch

VERSION = 'm2-4-audit-v2'
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
            raise ValueError('M2-4 artifact source differs: ' + key)
    if not torch.equal(meta['condition_mask'].cpu(), expected['condition_mask']):
        raise ValueError('M2-4 condition existence differs from current train')
    return rows, mapping


def validate_features(cache, cfg):
    if cache.get('implementation') != VERSION or 'metadata' not in cache:
        raise ValueError('old M2-4 cache has no auditable source; rebuild it')
    rows, mapping = validate_source(cache['metadata'], cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR, cfg.M2.CLIP_PATH)
    rgb = [row for row in rows if row[3] == 0]
    for key, values in [('pid', [r[1] for r in rgb]), ('cams', [r[2] for r in rgb]),
                        ('modality', [r[3] for r in rgb]), ('platform', [int(r[2] in (5, 6)) for r in rgb])]:
        if not torch.equal(cache[key].cpu(), torch.tensor(values)):
            raise ValueError('RGB feature row correspondence differs: ' + key)
    if cache['features'].shape != (len(rgb), 512) or not torch.isfinite(cache['features']).all():
        raise ValueError('invalid RGB teacher features')
    return mapping

def camera_balanced_centers(features, pids, cams, num_classes):
    features = torch.nn.functional.normalize(features.float(), dim=1)
    centers, camera_rows = [], []
    missing = [y for y in range(num_classes) if not (pids == y).any()]
    if missing:
        raise ValueError('train identities without RGB; no teacher centers fabricated: {}'.format(missing))
    for y in range(num_classes):
        per_camera = []
        for camera in cams[pids == y].unique(sorted=True):
            selected = (pids == y) & (cams == camera)
            per_camera.append(torch.nn.functional.normalize(features[selected].mean(0), dim=0))
            camera_rows.append((y, int(camera), int(selected.sum())))
        centers.append(torch.nn.functional.normalize(torch.stack(per_camera).mean(0), dim=0))
    return torch.stack(centers), camera_rows

def validate_centers(artifact, cfg, num_classes=None):
    if artifact.get('implementation') != VERSION or artifact.get('variant') != 'clip_relation_distill':
        raise ValueError('old/wrong M2-4 center artifact; rebuild it')
    rows, mapping = validate_source(artifact['metadata'], cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR, cfg.M2.CLIP_PATH)
    count = len(mapping)
    if num_classes is not None and count != num_classes:
        raise ValueError('teacher/student identity count differs')
    rgb = [row for row in rows if row[3] == 0]
    camera_counts = Counter((row[1], row[2]) for row in rgb)
    expected = sorted((y, c, count) for (y, c), count in camera_counts.items())
    if artifact['pid_order'] != list(range(count)) or artifact['camera_rows'] != expected:
        raise ValueError('M2-4 train PID/camera correspondence differs')
    if artifact.get('center_recipe') != 'image_l2_camera_mean_l2_equal_camera_mean_l2':
        raise ValueError('M2-4 teacher center recipe differs')
    centers = artifact['centers'].float()
    if centers.shape != (count, 512) or not torch.isfinite(centers).all():
        raise ValueError('invalid RGB centers')
    if not set(y for y, _, _ in expected) == set(range(count)):
        raise ValueError('train identity has no real RGB teacher center')
    torch.testing.assert_close(centers.norm(dim=1), torch.ones(count), rtol=1e-4, atol=1e-5)
    return count
