"""M2-5 source correspondence, shared P-ID text and stage metadata."""
import hashlib
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from config import cfg as defaults

VERSION = 'm2-5-audit-v2'
BASE_COMMIT = 'a8349abeb5fb62b5f7d233e33202d1ef1d113b42'
TEMPLATE = 'a photo of a X X X X person.'
PREPROCESS = {'size': [256, 128], 'interpolation': 'bilinear',
    'mean': [.48145466, .4578275, .40821073],
    'std': [.26862954, .26130258, .27577711],
    'ln_pre': True, 'activation': 'QuickGELU', 'use_c0': False}

def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_clip(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    return obj.state_dict() if isinstance(obj, torch.nn.Module) else obj

def code_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=str(Path(__file__).resolve().parents[2]), text=True).strip()

def load_config(path, opts=()):
    c = defaults.clone()
    c.merge_from_file(path)
    c.merge_from_list(list(opts))
    validate_config(c)
    return c

def validate_config(c):
    if not c.M2.ENABLED or c.M2.VARIANT != 'conditional_teacher_distill':
        raise ValueError('this worktree implements M2-5 conditional_teacher_distill')
    if (c.MODEL.TRANSFORMER_TYPE != 'vit_base_clip' or c.MODEL.TEXT_ALIGN or c.MODEL.VPR or
        c.MODEL.MOD_DELTA or c.MODEL.CE_SPLIT_VIEW or c.MODEL.CE_SPLIT_MODALITY or
        c.MODEL.SIE_CAMERA or c.MODEL.SIE_VIEW or c.MODEL.PE_FREEZE_BASE or
        c.MODEL.PE_TYPE != 'learnable' or c.MODEL.PE_LAYERWISE != 'none' or c.MODEL.DIST_TRAIN or
        c.MODEL.IF_LABELSMOOTH != 'off' or c.SOLVER.LOSS_TYPE != 'base' or
        c.DATALOADER.SYNC_FRAMES or c.DATALOADER.TIE_AUGMENTATION or
        any((c.SOLVER.TEXT_LOSS_WEIGHT, c.SOLVER.TWIN_LOSS_WEIGHT, c.SOLVER.TNCE_WEIGHT))):
        raise ValueError('M2-5 requires the independent original CE/Triplet/augmentation chain')
    if (list(c.INPUT.SIZE_TRAIN) != [256, 128] or list(c.INPUT.SIZE_TEST) != [256, 128] or
        list(c.MODEL.STRIDE_SIZE) != [16, 16] or c.MODEL.METRIC_LOSS_TYPE != 'triplet' or
        c.MODEL.IF_WITH_CENTER != 'no' or c.MODEL.LAYER_SCALE or c.MODEL.CHART_INPUT_NORM or
        list(c.INPUT.PIXEL_MEAN) != PREPROCESS['mean'] or list(c.INPUT.PIXEL_STD) != PREPROCESS['std']):
        raise ValueError('M2-5 first-launch geometry/original CLIP/Triplet contract differs')
    if list(c.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal'] or list(c.DATASETS.AERIAL_CAMS) != [5, 6]:
        raise ValueError('M2-5 modality/platform mapping differs')
    if c.M2.TRAIN_STAGE not in ('student', 'teacher'):
        raise ValueError('unknown M2-5 training stage')
    if c.M2.TRAIN_STAGE == 'student' and (not c.MODEL.TOKEN_TRAJECTORY or
        c.MODEL.TOKEN_TRAJECTORY_VARIANT != 'dense' or c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX != 0.):
        raise ValueError('M2-5 first student launch requires original velocity-only dense C0')
    if (c.TEACHER_STAGE.USE_C0 or not c.TEACHER_STAGE.FREEZE_PRETRAINED_BACKBONE or
        list(c.TEACHER_STAGE.LORA_TARGETS) != ['q', 'v'] or
        c.TEACHER_STAGE.MODALITY_BANKS != 3 or c.TEACHER_STAGE.PLATFORM_BANKS != 2 or
        not c.M2.TEACHER_SAME_AUGMENTED_INPUT):
        raise ValueError('M2-5 requires frozen no-C0 Q/V teacher and identical augmented inputs')
    if c.TEACHER_STAGE.TEMPERATURE != c.M2.TEXT_TEMPERATURE:
        raise ValueError('teacher/student base text temperatures must match')
    if c.MODEL.PRETRAIN_CHOICE == 'imagenet' and Path(c.MODEL.PRETRAIN_PATH).resolve() != Path(c.M2.CLIP_PATH).resolve():
        raise ValueError('M2-5 student/teacher must use the same original CLIP')

def train_source(c):
    import re
    rows = []
    root = Path(c.DATASETS.ROOT_DIR) / (c.DATASETS.SUBDIR or 'WHU-MARS') / 'train'
    for m, name in enumerate(c.DATASETS.MODALITIES):
        for path in sorted((root / name).glob('*.jpg')):
            raw, camera = map(int, re.search(r'(\d+)_c(\d+)', path.name).groups())
            rows.append((str(path), raw, camera - 1, m))
    if not rows:
        raise ValueError('no train images found')
    mapping = {pid: y for y, pid in enumerate(sorted({r[1] for r in rows}))}
    rows = [(p, mapping[y], c0, m) for p, y, c0, m in rows]
    mask = torch.zeros(len(mapping), 3, 2, dtype=torch.bool)
    for _, y, cam, m in rows:
        mask[y, m, int(cam in (5, 6))] = True
    meta = {'split': 'train', 'pid_map': mapping,
        'filenames': [c.DATASETS.MODALITIES[r[3]] + '/' + Path(r[0]).name for r in rows],
        'condition_mask': mask, 'modality_names': list(c.DATASETS.MODALITIES),
        'clip_sha256': file_digest(c.M2.CLIP_PATH), 'preprocess': PREPROCESS.copy()}
    return rows, meta

def validate_source(meta, c):
    rows, expected = train_source(c)
    for key in ('split', 'pid_map', 'filenames', 'modality_names', 'clip_sha256', 'preprocess'):
        if meta.get(key) != expected[key]:
            raise ValueError('M2-5 source differs: ' + key)
    if not torch.equal(meta['condition_mask'].cpu(), expected['condition_mask']):
        raise ValueError('M2-5 condition mask differs')
    return rows, expected

def validate_cache(obj, c):
    if obj.get('implementation') != VERSION:
        raise ValueError('old/unverified M2-5 cache; rebuild it')
    rows, source = validate_source(obj['metadata']['source'], c)
    for key, values in [('pids', [r[1] for r in rows]), ('camids', [r[2] for r in rows]),
        ('modalities', [r[3] for r in rows]), ('platforms', [int(r[2] in (5, 6)) for r in rows])]:
        if not torch.equal(obj[key].cpu(), torch.tensor(values)):
            raise ValueError('M2-5 cache row correspondence differs: ' + key)
    if obj['clip_features'].shape != (len(rows), 512) or not torch.isfinite(obj['clip_features']).all():
        raise ValueError('invalid M2-5 CLIP features')
    torch.testing.assert_close(obj['clip_features'].norm(dim=1), torch.ones(len(rows)), rtol=1e-4, atol=1e-5)
    return source

def template_tokens(device='cpu'):
    from model.backbones.clip_text import build_tokenizer
    tok = build_tokenizer()
    seq = [49406] + tok.encode(TEMPLATE) + [49407]
    placeholder = tok.encode('X')
    if len(placeholder) != 1:
        raise ValueError('X must tokenize to one token')
    slots = [i for i, token in enumerate(seq) if token == placeholder[0]]
    if len(slots) != 4:
        raise ValueError('P-ID needs four real token slots')
    return torch.tensor(seq + [0] * (77 - len(seq)), device=device), slots, len(seq) - 1

def validate_text_bank(bank, c, num_classes=None):
    meta = bank['metadata']
    if meta.get('implementation') != VERSION or meta.get('stage') != 'prompt':
        raise ValueError('old/wrong M2-5 text bank; rebuild P-ID')
    _, source = validate_source(meta['source'], c)
    _, slots, eot = template_tokens()
    if (meta['template'] != TEMPLATE or meta['token_slots'] != slots or meta['eot_index'] != eot or
        meta['optimizer_updates'] != c.M2.PROMPT_STEPS or meta['temperature'] != c.M2.TEXT_TEMPERATURE):
        raise ValueError('M2-5 text template/budget/temperature differs')
    y = len(source['pid_map'])
    if num_classes is not None and num_classes != y:
        raise ValueError('M2-5 train identity count differs')
    text = bank['text_bank'].float()
    if text.shape != (y, 512) or not torch.isfinite(text).all():
        raise ValueError('invalid shared P-ID text bank')
    torch.testing.assert_close(text.norm(dim=1), torch.ones(y), rtol=1e-4, atol=1e-5)
    if meta['prompt_sha256'] != file_digest(c.M2.PROMPT_WEIGHT):
        raise ValueError('M2-5 text bank links another final prompt')
    prompt = torch.load(c.M2.PROMPT_WEIGHT, map_location='cpu', weights_only=False)
    if (prompt['tokens'].shape != (y, 4, 512) or not torch.isfinite(prompt['tokens']).all() or
        prompt['metadata']['optimizer_updates'] != meta['optimizer_updates'] or
        not torch.equal(prompt['text_bank'].float(), text)):
        raise ValueError('M2-5 text table differs from the final ordinary P-ID export')
    validate_source(prompt['metadata']['source'], c)
    return source

def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(), 'python': random.getstate()}

def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])

def seed_stage(c):
    from train import set_seed
    set_seed(c.SOLVER.SEED, c.SOLVER.CUDNN_BENCHMARK)
