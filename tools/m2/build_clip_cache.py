"""Build a train-only deterministic CLIP image cache for M2-5."""
import argparse,json,re
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model.backbones.vit_pytorch import vit_base_clip
PAT=re.compile(r"(\d+)_c(\d+)")
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--clip',required=True);p.add_argument('--out',required=True);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='cuda');a=p.parse_args(); root=Path(a.data_root)/'train';mods=['RGB','IR','Thermal'];rows=[]
 for mi,m in enumerate(mods):
  for f in sorted((root/m).glob('*.jpg')):
   q=PAT.search(f.name)
   if q:
    pid,cam=map(int,q.groups());rows.append((str(f),pid,cam-1,mi,m))
 pids={x:i for i,x in enumerate(sorted({r[1] for r in rows}))};tf=transforms.Compose([transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))]);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');net=vit_base_clip(img_size=(256,128),stride_size=16).to(dev).eval();net._load_clip_visual(torch.load(a.clip,map_location='cpu'));[x.requires_grad_(False) for x in net.parameters()];out=[]
 with torch.no_grad():
  for i in range(0,len(rows),a.batch_size): out.append(net(torch.stack([tf(Image.open(r[0]).convert('RGB')) for r in rows[i:i+a.batch_size]]).to(dev)).float().cpu())
 feat=torch.cat(out);z=torch.nn.functional.normalize(feat@net.clip_proj.detach().cpu(),dim=-1);payload={'features':z,'pids':torch.tensor([pids[r[1]] for r in rows]),'raw_pids':torch.tensor([r[1] for r in rows]),'camids':torch.tensor([r[2] for r in rows]),'modalities':torch.tensor([r[3] for r in rows]),'platforms':torch.tensor([1 if r[2] in (5,6) else 0 for r in rows]),'paths':[r[0] for r in rows],'clip_path':str(Path(a.clip).resolve()),'preprocess':{'size':[256,128],'mean':[.48145466,.4578275,.40821073],'std':[.26862954,.26130258,.27577711]},'pid_map':pids};Path(a.out).parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.out);print(json.dumps({'rows':len(rows),'pids':len(pids),'out':str(Path(a.out).resolve())},ensure_ascii=False))
if __name__=='__main__':main()
