"""Build M2-4 train-only RGB identity centers from original CLIP images."""
import argparse,re
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model.backbones.vit_pytorch import vit_base_clip
from tools.m2_cache import build_relation_cache
PAT=re.compile(r'(\d+)_c(\d+)')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--clip',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='cuda');a=p.parse_args(); root=Path(a.data_root)/'train'/'RGB'; rows=[]
 for f in sorted(root.glob('*.jpg')):
  m=PAT.search(f.name)
  if m: rows.append((f,int(m.group(1)),int(m.group(2))-1))
 pid_order=sorted({r[1] for r in rows}); pid_map={p:i for i,p in enumerate(pid_order)}; dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); tf=transforms.Compose([transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))]); net=vit_base_clip(img_size=(256,128),stride_size=16,drop_path_rate=0.).to(dev).eval(); obj=torch.load(a.clip,map_location='cpu',weights_only=False); net._load_clip_visual(obj.state_dict() if isinstance(obj,torch.nn.Module) else obj); [q.requires_grad_(False) for q in net.parameters()]; feats=[]
 with torch.no_grad():
  for i in range(0,len(rows),a.batch_size): feats.append((net(torch.stack([tf(Image.open(r[0]).convert('RGB')) for r in rows[i:i+a.batch_size]]).to(dev)).float() @ net.clip_proj.detach()).cpu())
 z=torch.nn.functional.normalize(torch.cat(feats),dim=-1); sums={}; counts={}
 for i,(_,raw,cam) in enumerate(rows): sums.setdefault((raw,cam),torch.zeros(512)); sums[(raw,cam)]+=z[i]; counts[(raw,cam)]=counts.get((raw,cam),0)+1
 centers=[]
 for raw in pid_order:
  cams=sorted(c for (p,c) in sums if p==raw); per=[torch.nn.functional.normalize(sums[(raw,c)]/counts[(raw,c)],dim=0) for c in cams]; centers.append(torch.nn.functional.normalize(torch.stack(per).mean(0),dim=0))
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); payload={'centers':torch.stack(centers),'pid_order':list(range(len(pid_order))),'metadata':{'split':'train','modality':'RGB','feature_dim':512,'normalize':'l2','raw_pid_order':pid_order,'clip_path':str(Path(a.clip).resolve()),'preprocess':{'size':[256,128],'mean':[.48145466,.4578275,.40821073],'std':[.26862954,.26130258,.27577711]}}}; torch.save(payload,out); build_relation_cache(payload['centers'],payload['pid_order'],str(out.with_name('clip_identity_relation.pt')),metadata=payload['metadata']); print('saved',out)
if __name__=='__main__': main()
