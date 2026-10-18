"""Exercise the real 192-image production loop until two effective AMP updates."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
assert not Path(args.output).exists(), 'preserve earlier preflight output'
updates = 0
original_step = torch.optim.Adam.step

class PreflightComplete(Exception):
    pass

def step(self, *a, **kw):
    global updates
    result = original_step(self, *a, **kw)
    updates += 1
    if updates == 2:
        raise PreflightComplete()
    return result

torch.optim.Adam.step = step
sys.argv = [str(ROOT/'finetune.py'), '--config_file', args.config, 'OUTPUT_DIR', args.output]
try:
    runpy.run_path(str(ROOT/'finetune.py'), run_name='__main__')
    raise RuntimeError('Production loop ended before two updates')
except PreflightComplete as completed:
    trace = completed.__traceback__
    frame = None
    while trace is not None:
        if trace.tb_frame.f_code.co_name == 'do_train':
            frame = trace.tb_frame
        trace = trace.tb_next
    assert frame is not None
    model = frame.f_locals['model_meta']
    config = frame.f_locals['cfg']
    images = frame.f_locals['imgs']
    cameras = frame.f_locals['camids']
    assert sum(len(x) for x in images) == 192
    teacher = frame.f_locals['m2_teacher']
    if config.M2.ENABLED:
        assert teacher is not None and not teacher.training
        assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
        assert getattr(teacher.base, 'history_adapter', None) is None
        assert teacher.base.token_trajectory is None
        assert not model.base.clip_proj.requires_grad
        kd = float(frame.f_locals['m5_aux'].detach())
        assert kd > 0
    else:
        assert teacher is None
        kd = 0.
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all(), name
    # Use the real deployment loader, including M2/History/variant metadata.
    from model import make_model
    snapshot = Path(args.output)/'preflight_student.pth'
    payload = {'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()}}
    if config.M2.ENABLED:
        payload['metadata'] = model.m5_metadata
    torch.save(payload, snapshot)
    copy_config = config.clone()
    copy_config.defrost()
    copy_config.MODEL.PRETRAIN_CHOICE = 'no'
    restored = make_model(copy_config, 500, 7, 1)
    restored.load_param(str(snapshot))
    for key, value in restored.state_dict().items():
        if not key.startswith('classifier.'):
            assert torch.equal(value, payload['state_dict'][key]), key
    restored.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        feature = model(images[0][:2], camids=cameras[0][:2], mode=1)
        assert torch.isfinite(feature).all()
    snapshot.unlink()
    record = dict(status='PRODUCTION_PREFLIGHT_OK', effective_amp_updates=updates,
                  batch_images=192, kd_loss=kd, deployment_visual_reload_equal=True,
                  strict_state_reload=True,
                  teacher_disabled_adapters=True,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    Path(args.output,'preflight.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
