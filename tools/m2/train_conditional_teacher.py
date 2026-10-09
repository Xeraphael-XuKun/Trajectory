"""M2-5 stage B: train only conditional Q/V-LoRA, BN and classifier."""
import argparse, random, time, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model.backbones.vit_pytorch import vit_base_clip
from model.m2_teacher import ConditionalFeatureTeacher
from tools.m2.artifacts import read_clip, file_digest, rng_state

def hard_triplet(g,y):
    d=torch.cdist(g.float(),g.float()); pos=(y[:,None]==y[None,:]); neg=~pos; pos.fill_diagonal_(False); dp=d.masked_fill(~pos,-1).max(1).values; dn=d.masked_fill(~neg,1e6).min(1).values; return F.softplus(dp-dn).mean()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--text-bank',required=True); p.add_argument('--clip',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--lr',type=float,default=3.5e-4); p.add_argument('--device',default='cuda'); a=p.parse_args()
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); cache=torch.load(a.cache,map_location='cpu',weights_only=False); bank=torch.load(a.text_bank,map_location='cpu',weights_only=False); paths=cache['paths']; yall=cache['pids'].long(); mods=cache['modalities'].long(); plats=cache['platforms'].long(); text_bank=bank['text_bank'].float().to(dev); proj=cache.get('clip_proj');
 if proj is None or cache.get('split')!='train': raise ValueError('teacher cache must be train-only and contain clip_proj')
 proj=proj.float().to(dev); backbone=vit_base_clip(img_size=(256,128),stride_size=16,drop_path_rate=.1).to(dev); backbone._load_clip_visual(read_clip(a.clip)); backbone.train();
 for q in backbone.parameters(): q.requires_grad_(False)
 teacher=ConditionalFeatureTeacher(dim=768,modalities=3,platforms=2,rank=8,alpha=8).to(dev); teacher.classifier=nn.Linear(768,int(yall.max())+1).to(dev); teacher.bn=nn.BatchNorm1d(768).to(dev); teacher.train(); params=[q for q in teacher.parameters() if q.requires_grad]; opt=torch.optim.Adam(params,lr=a.lr,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); scaler=torch.cuda.amp.GradScaler(enabled=dev.type=='cuda'); tf=transforms.Compose([transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BICUBIC),transforms.RandomHorizontalFlip(.5),transforms.Pad(10),transforms.RandomCrop((256,128)),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))]); g=torch.Generator().manual_seed(1234); start=time.time()
 by_pm={(int(y),int(m)):torch.where((yall==y)&(mods==m))[0].tolist() for y in yall.unique().tolist() for m in range(3)}
 ids_all=yall.unique().tolist(); steps=max(1,len(paths)//(16*4*3))
 for ep in range(a.epochs):
  for _ in range(steps):
   chosen=random.sample(ids_all,min(16,len(ids_all))); take=[]
   for yy in chosen:
    for mm in range(3):
     pool=by_pm.get((yy,mm),[])
     if not pool: continue
     take.extend(random.choices(pool,k=4) if len(pool)<4 else random.sample(pool,4))
   take=torch.tensor(take,dtype=torch.long); imgs=torch.stack([tf(Image.open(paths[int(i)]).convert('RGB')) for i in take]).to(dev); y=yall[take].to(dev); m=mods[take].to(dev); v=plats[take].to(dev)
   with torch.cuda.amp.autocast(enabled=dev.type=='cuda'):
    gT=teacher.forward_visual(backbone,imgs,m,v); fT=teacher.bn(gT); z=F.normalize(gT.float()@proj,dim=-1); loss=F.cross_entropy(teacher.classifier(fT),y)+hard_triplet(gT,y)+.5*F.cross_entropy(z@text_bank.t()/.07,y)
   opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,1.); scaler.step(opt); scaler.update()
  sched.step(); print('teacher_epoch=%d loss=%.6f'%(ep+1,float(loss.detach())))
 state={'state_dict':teacher.state_dict(),'epochs':a.epochs,'temperature':.07,'text_bank':str(Path(a.text_bank).resolve()),'clip_path':str(Path(a.clip).resolve()),'clip_digest':file_digest(a.clip),'dim':768,'lora_targets':['q','v'],'split':'train','optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'budget':{'seconds':time.time()-start,'epochs':a.epochs}}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save(state,a.out)
if __name__=='__main__': main()
