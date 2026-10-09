import sys,json,gc,argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
p=argparse.ArgumentParser(description='CPU 核验 HI 最终权重和局部适配器开关；不重训，不替代完整检索。')
p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
p.add_argument('--runs-root',required=True)
p.add_argument('--data-root',required=True)
p.add_argument('--output',required=True)
p.add_argument('--dependency-path',default='')
args=p.parse_args()
REPO=Path(args.repo)
sys.path.insert(0,str(REPO))
if args.dependency_path:sys.path.insert(0,args.dependency_path)
from config import cfg
from model import make_model
torch.set_num_threads(4)
out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
root=Path(args.runs_root)/'history_innovation_1009'
files=[]
for mod in ['RGB','IR','Thermal']:
 files.extend(sorted((Path(args.data_root)/'query'/mod).glob('*.jpg'))[:4])
assert len(files)==12
images=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize((128,256),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255 for p in files])
images=(images-torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None])/torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None]
report={'local_runtime':torch.__version__,'device':'cpu','image_paths':[str(p) for p in files],'preprocess':'PIL bicubic 256x128, CLIP normalization; local probe, not full evaluation','groups':{}}
coordinate_reference=None
for name in ['HI0','HI1','HI2','HI3']:
 state=torch.load(root/(name+'_seed1234')/'transformer_60.pth',map_location='cpu')
 c=cfg.clone();c.merge_from_file(str(REPO/'configs'/('history_'+name+'.yml')));c.MODEL.PRETRAIN_CHOICE='no'
 model=make_model(c,500,7,2).eval()
 init={k:v.detach().clone() for k,v in model.state_dict().items() if 'history_adapter' in k}
 model.load_state_dict(state,strict=True)
 r={'keys':len(state),'all_finite':all(bool(torch.isfinite(v).all()) for v in state.values()),'specification':state['base.history_adapter.specification'].tolist(),'bn_batches':{k:int(v) for k,v in state.items() if 'num_batches_tracked' in k},'families':{}}
 for family in ['gain','cls_gain','anchors','correctors','predictors']:
  keys=[k for k in init if k.startswith('base.history_adapter.'+family)]
  if not keys:continue
  v=torch.cat([state[k].flatten().double() for k in keys]);v0=torch.cat([init[k].flatten().double() for k in keys])
  r['families'][family]={'numel':v.numel(),'rms_double':v.square().mean().sqrt().item(),'abs_max':v.abs().max().item(),'initial_rms':v0.square().mean().sqrt().item(),'fraction_abs_lt_1e-20':(v.abs()<1e-20).double().mean().item(),'fraction_exact_zero':(v==0).double().mean().item()}
 p=state['base.history_adapter.coordinate']
 if coordinate_reference is None:coordinate_reference=p.clone()
 r['coordinate_equals_HI0']=torch.equal(p,coordinate_reference)
 r['orthogonality_max_error']=(p.T@p-torch.eye(64)).abs().max().item()
 adapter=model.base.history_adapter
 r['forward_probe']={}
 for neck in ['after','before']:
  model.neck_feat=neck
  with torch.no_grad(): enabled=model(images,mode=1)
  model.base.history_adapter=None
  with torch.no_grad(): disabled=model(images,mode=1)
  model.base.history_adapter=adapter
  r['forward_probe'][neck]={'exact_equal':torch.equal(enabled,disabled),'max_abs_diff':(enabled-disabled).abs().max().item(),'shape':list(enabled.shape)}
 report['groups'][name]=r
 print(name,json.dumps(r),flush=True)
 del model,state,init,adapter;gc.collect()
(out/'checkpoint_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
