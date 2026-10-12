"""Read-only stage-two diagnostics; use the parent project's model and metrics."""
import csv
import hashlib
import inspect
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
ORIGINAL_CPU_THREADS = torch.get_num_threads()
torch.set_num_threads(2)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FORMAL = False
DIAGNOSTIC_VERSION = 'r2_diagnostics_v2'

# Only this diagnostic process supports the local torch 1.13 loader. Model math,
# strict state loading and original source/teacher validators are untouched.
if 'weights_only' not in inspect.signature(torch.load).parameters:
    _torch_load = torch.load
    def _load_legacy(*args, **kwargs):
        kwargs.pop('weights_only', None)
        return _torch_load(*args, **kwargs)
    torch.load = _load_legacy


def load(path):
    return torch.load(str(path), map_location='cpu', weights_only=False)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def environment():
    import timm, torchvision
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, torchvision=torchvision.__version__, timm=timm.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                diagnostic_version=DIAGNOSTIC_VERSION,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cpu_threads=torch.get_num_threads(),
                loader_compatibility='legacy keyword adapter' if torch.__version__.startswith('1.13.') else 'original torch.load')


def execution_mode(formal):
    global FORMAL
    FORMAL = formal
    if formal:
        torch.set_num_threads(ORIGINAL_CPU_THREADS)
        env = environment()
        expected = dict(torch='2.4.1+cu121', torchvision='0.13.1', timm='1.0.19')
        for key, value in expected.items():
            assert env[key].split('+')[0] == value.split('+')[0], (key, env[key], value)
        assert sys.version_info[:2] == (3, 8), sys.version
        assert torch.version.cuda == '12.1', torch.version.cuda
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        assert os.environ.get('WORLD_SIZE') == '1'


def code_commit():
    import subprocess
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()


def resources(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config_from_text(text, paths, role='m5'):
    from config import cfg
    from yacs.config import CfgNode
    # These are complete saved configs. Re-merging them would make YACS decode
    # the already-typed DEVICE_ID string '0' into an int, unlike file loading.
    c = CfgNode.load_cfg(text)
    if role == 'c0':
        # Logger str(cfg) omits YAML quotes, unlike checkpoint cfg.dump().
        # Recover original string fields such as 'yes', 'on' and device '0'.
        import yaml
        literal = yaml.load(text, Loader=yaml.BaseLoader)
        def restore_strings(current, default, raw):
            for key, value in current.items():
                if key not in default:
                    continue
                if isinstance(value, CfgNode):
                    restore_strings(value, default[key], raw[key])
                elif isinstance(default[key], str) and not isinstance(value, str):
                    current[key] = raw[key]
        restore_strings(c, cfg, literal)
    def add_missing(current, default):
        for key, value in default.items():
            if key not in current:
                current[key] = value.clone() if isinstance(value, CfgNode) else value
            elif isinstance(value, CfgNode):
                add_missing(current[key], value)
    add_missing(c, cfg)
    if isinstance(c.DATASETS.MODALITIES, str):
        import ast
        c.DATASETS.MODALITIES = ast.literal_eval(c.DATASETS.MODALITIES)
    c.DATASETS.ROOT_DIR = str(Path(paths['dataset']).parent)
    c.DATASETS.SUBDIR = Path(paths['dataset']).name
    c.MODEL.PRETRAIN_PATH = paths['clip']
    c.MODEL.TEXT_CLIP_PATH = paths['clip']
    c.M2.CLIP_PATH = paths['clip']
    if role == 'm5':
        c.M2.TEACHER_WEIGHT = paths['m5_teacher']
        c.M2.TEXT_BANK = paths['m5_text']
        c.M2.PROMPT_WEIGHT = paths['m5_prompt']
    else:
        c.M2.ENABLED = False
    return c


def m5_config(paths):
    return config_from_text(load(paths['m5_student'])['metadata']['config'], paths)


def c0_config(paths):
    return config_from_text(c0_historical_config(paths), paths, 'c0')


def c0_historical_config(paths):
    text = Path(paths['c0_train_log']).read_text(encoding='utf-8')
    text = text.split('Running with config:\n', 1)[1]
    text = re.split(r'\n\d{4}-\d\d-\d\d ', text, 1)[0]
    return text


def student(paths, device, role='m5'):
    from model import make_model
    c = m5_config(paths) if role == 'm5' else c0_config(paths)
    c.MODEL.PRETRAIN_CHOICE = 'no'  # every state tensor is strictly restored below
    obj = load(paths['m5_student' if role == 'm5' else 'c0_student'])
    state = obj.get('state_dict', obj)
    if role == 'm5':
        meta = obj['metadata']
        assert meta['stage'] == 'student' and meta['epoch'] == 60
        assert meta['code_commit'] == paths['m5_parent']
        assert meta['teacher_sha256'] == digest(paths['m5_teacher'])
        assert meta['text_sha256'] == digest(paths['m5_text'])
    model = make_model(c, 500, 7, 3)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), c


def teacher(paths, device):
    from utils.m5_checkpoint import load_teacher, architecture
    c = m5_config(paths)
    try:
        model, obj = load_teacher(c, 500, device)
    except ValueError as exc:
        if FORMAL or str(exc) != 'M2-5 teacher frozen original visual changed: pos_embed' or not torch.__version__.startswith('1.13.'):
            raise
        # Explicit diagnostic-only compatibility: all original checks remain,
        # except tiny recomputed pos_embed rounding. The stored tensor itself
        # is loaded exactly, not replaced by the locally interpolated tensor.
        from tools.m2.artifacts import VERSION, validate_text_bank, validate_source
        from model.m2_teacher import ConditionalVisualTeacher
        obj = load(paths['m5_teacher'])
        meta = obj['metadata']
        assert meta['implementation'] == VERSION and meta['variant'] == c.M2.VARIANT
        assert meta['stage'] == 'teacher' and meta['epoch'] == c.TEACHER_STAGE.MAX_EPOCHS
        assert meta['teacher_architecture'] == architecture(c)
        assert meta['text_sha256'] == digest(paths['m5_text'])
        validate_text_bank(load(paths['m5_text']), c, 500)
        validate_source(meta['source'], c)
        model = ConditionalVisualTeacher(c, 500, load_original=True)
        differences = {}
        for key, value in model.base.state_dict().items():
            if key.startswith('fc.'):
                continue
            stored = obj['state_dict']['base.'+key]
            if torch.equal(value, stored):
                continue
            assert key == 'pos_embed', key
            delta = (value-stored).float()
            maximum, rms = delta.abs().max().item(), delta.square().mean().sqrt().item()
            assert maximum <= 2e-8 and rms <= 2e-9, (maximum, rms)
            differences[key] = dict(max_abs=maximum, rms=rms,
                reason='local recomputation differs; torch1.13 vs historical2.4 interpolation is a candidate cause',
                stored_tensor_loaded_exactly=True)
        model.load_state_dict(obj['state_dict'], strict=True)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.diagnostic_compatibility = differences
        print('DIAGNOSTIC_COMPATIBILITY', differences, flush=True)
    return model, obj['metadata']


def index_split(paths, split):
    rows = []
    root = Path(paths['dataset'])
    for mod, name in enumerate(('RGB', 'IR', 'Thermal')):
        for p in sorted((root / split / name).glob('*.jpg')):
            pid, camera = map(int, re.search(r'(\d+)_c(\d+)', p.name).groups())
            rows.append(dict(index=len(rows), path=str(p), relative_path=p.relative_to(root).as_posix(),
                             raw_pid=pid, camid=camera-1, modality=mod,
                             modality_name=name, platform=int(camera-1 in (5, 6))))
    if split == 'train':
        mapping = {p: y for y, p in enumerate(sorted({r['raw_pid'] for r in rows}))}
        for r in rows:
            r['train_pid'] = mapping[r['raw_pid']]
    return rows


def transforms(c, augmented=False):
    import torchvision.transforms as T
    if augmented:
        from timm.data.random_erasing import RandomErasing
        return T.Compose([T.Resize(c.INPUT.SIZE_TRAIN, interpolation=3),
                          T.RandomHorizontalFlip(p=c.INPUT.PROB), T.Pad(c.INPUT.PADDING),
                          T.RandomCrop(c.INPUT.SIZE_TRAIN), T.ToTensor(),
                          T.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD),
                          RandomErasing(probability=c.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')])
    return T.Compose([T.Resize(c.INPUT.SIZE_TEST), T.ToTensor(),
                      T.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD)])


class Images(torch.utils.data.Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.rows[i]['path']) as im:
            return self.transform(im.convert('RGB')), i


def seed_diagnostics(seed=1234, formal=False, config=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if formal:
        from utils.runtime import configure_cudnn
        configure_cudnn(config.SOLVER.CUDNN_BENCHMARK)
        # Preserve torch 2.4's original TF32 defaults, as in historical test.py.
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def buffer_snapshot(model):
    return {n: t.detach().cpu().clone() for n, t in model.named_buffers()}


def buffers_equal(model, snapshot):
    return all(torch.equal(t.cpu(), snapshot[n]) for n, t in model.named_buffers())
