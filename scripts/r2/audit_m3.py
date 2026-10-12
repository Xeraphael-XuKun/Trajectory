"""Run M3's own bank/source validators without importing M5's model package."""
import argparse
import subprocess
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--resources', required=True)
ap.add_argument('--output', required=True)
args = ap.parse_args()
paths = resources(args.resources)
source_files=('config/defaults.py','model/make_model.py','model/m2/deep_text_control.py',
              'utils/m3_artifacts.py','solver/make_optimizer.py')
for file in source_files:
    original=subprocess.check_output(['git','-C',paths['m3_repo'],'show',paths['m3_parent']+':'+file])
    current=(Path(paths['m3_repo'])/file).read_bytes()
    assert current.replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n'), 'M3 parent source differs: '+file
sys.path.insert(0, paths['m3_repo'])
from config import cfg
from yacs.config import CfgNode
from utils.m3_artifacts import validate_bank
from model.m2.deep_text_control import DeepIdentityText
obj = load(paths['m3_student'])
meta = obj['metadata']
assert meta['epoch'] == 60 and meta['code_commit'] == paths['m3_parent']
c = CfgNode.load_cfg(meta['config'])
c.DATASETS.ROOT_DIR = str(Path(paths['dataset']).parent)
c.DATASETS.SUBDIR = Path(paths['dataset']).name
c.M2.CLIP_PATH = paths['clip']
bank = load(paths['m3_bank'])
assert digest(paths['m3_bank']) == meta['prompt_sha256']
# The recorded signature is compared with the real class/tokenizer signature.
clip = load(c.M2.CLIP_PATH)
text = DeepIdentityText(clip.state_dict() if isinstance(clip, torch.nn.Module) else clip, 500)
validate_bank(bank, c, 500, text.signature)
out = Path(args.output).parent
(out/'config_m3_historical.yml').write_text(meta['config'], encoding='utf-8')
c.MODEL.PRETRAIN_PATH = paths['clip']
c.MODEL.TEXT_CLIP_PATH = paths['clip']
c.M2.TEXT_BANK = paths['m3_bank']
for field in ('ID_PROMPT_WEIGHT','DEEP_CONTEXT_INIT'):
    if getattr(c.M2,field):
        setattr(c.M2,field,paths['m3_bank'])
(out/'config_m3_local.yml').write_text(c.dump(), encoding='utf-8')
del text, clip
c.M2.DEPLOY_ONLY = True
c.MODEL.PRETRAIN_CHOICE = 'no'
from model import make_model
from solver.make_optimizer import make_optimizer
model = make_model(c,500,7,3).eval()
model.load_state_dict(obj['state_dict'],strict=True)
optimizer,_ = make_optimizer(c,model,torch.nn.Linear(1,1))
names = {id(p):n for n,p in model.named_parameters()}
groups = [dict(index=i,parameters=[names[id(p)] for p in g['params']],
               numel=sum(p.numel() for p in g['params']),lr=g['lr'],weight_decay=g['weight_decay'])
          for i,g in enumerate(optimizer.param_groups)]
controls=[g for g in groups if any('deep_c0_controller' in n for n in g['parameters'])]
assert len(controls)==5 and sum(g['numel'] for g in controls)==186368
assert all(g['lr']==.00035 and g['weight_decay']==.0001 for g in controls)
write_json(out/'optimizer_groups_m3_parent.json',groups)
write_json(args.output, dict(status='PASS', parent=meta['code_commit'], prompt_hash_matches=True,
                            parent_source_files_verified=list(source_files),
                            source_pid_mapping_verified=True, signature_verified=True,
                            deployment_strict_load=True,controller_groups=controls,
                            context_init_shape=list(bank['context_init'].shape)))
