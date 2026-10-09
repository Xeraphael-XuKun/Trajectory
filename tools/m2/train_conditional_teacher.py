"""Train the M2-5 frozen-CLIP conditional Q/V-LoRA teacher."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model.backbones.vit_pytorch import vit_base_clip
from model.m2_teacher import ConditionalFeatureTeacher

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--text-bank',required=True); p.add_argument('--clip',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--lr',type=float,default=3.5e-4); p.add_argument('--device',default='cuda'); a=p.parse_args()
    dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False); bank=torch.load(a.text_bank,map_location='cpu',weights_only=False)
    paths=cache['paths']; pids=cache['pids'].long(); mods=cache['modalities'].long(); plats=cache['platforms'].long(); text_bank=bank['text_bank'].float().to(dev); proj=cache.get('clip_proj')
    if proj is None: raise ValueError('cache must contain clip_proj for teacher text supervision')
    proj=proj.float().to(dev)
    backbone=vit_base_clip(img_size=(256,128),stride_size=16).to(dev); backbone._load_clip_visual(torch.load(a.clip,map_location='cpu',weights_only=False)); backbone.eval()
    for q in backbone.parameters(): q.requires_grad_(False)
    teacher=ConditionalFeatureTeacher(dim=768,modalities=3,platforms=2,rank=8,alpha=8).to(dev); teacher.classifier=torch.nn.Linear(768,int(pids.max())+1).to(dev); teacher.train()
    opt=torch.optim.Adam([q for q in teacher.parameters() if q.requires_grad],lr=a.lr,weight_decay=1e-4)
    tf=transforms.Compose([transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))]); g=torch.Generator().manual_seed(1234)
    for ep in range(a.epochs):
        order=torch.randperm(len(paths),generator=g)
        for start in range(0,len(paths),a.batch_size):
            take=order[start:start+a.batch_size]; images=torch.stack([tf(Image.open(paths[int(i)]).convert('RGB')) for i in take]).to(dev); y=pids[take].to(dev); m=mods[take].to(dev); v=plats[take].to(dev)
            h=teacher.forward_visual(backbone,images,m,v); z=F.normalize(h.float()@proj,dim=-1); loss=F.cross_entropy(teacher.classifier(h),y)+.5*F.cross_entropy(z@text_bank.t()/.07,y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
        print('teacher_epoch=%d loss=%.6f'%(ep+1,float(loss.detach())))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':teacher.state_dict(),'epochs':a.epochs,'temperature':.07,'text_bank':str(Path(a.text_bank).resolve()),'clip_path':str(Path(a.clip).resolve()),'dim':768,'lora_targets':['q','v'],'split':'train'},a.out)
if __name__=='__main__': main()
