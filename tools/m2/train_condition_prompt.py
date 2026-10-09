"""Train M2-1 condition text prototypes from a verified train-only CLIP cache."""
import argparse, json, torch
from torch.nn import functional as F
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer, SOT, EOT
from model.make_model import _read_clip_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',required=True); ap.add_argument('--clip',required=True); ap.add_argument('--output',required=True); ap.add_argument('--steps',type=int,default=10000); args=ap.parse_args()
    c=torch.load(args.cache,map_location='cpu',weights_only=False); feat=F.normalize(c['feature'].float(),dim=1); pid=torch.as_tensor(c['pid']); mod=torch.as_tensor(c['modality']); plat=torch.as_tensor(c['platform']); n=int(pid.max())+1
    sd=_read_clip_checkpoint(args.clip); text=CLIPTextEncoder(); text.load_clip(sd); tok=build_tokenizer(); base=text.token_embedding.weight.detach(); S=torch.nn.Parameter(torch.randn(n,4,512)*.02); U=torch.nn.Parameter(torch.randn(3,2,512)*.02); Q=torch.nn.Parameter(torch.randn(2,2,512)*.02); opt=torch.optim.Adam([S,U,Q],lr=3.5e-4,weight_decay=1e-4)
    rows={}
    for step in range(args.steps):
        m=int(torch.randint(0,3,()).item()); v=int(torch.randint(0,2,()).item()); avail=((mod==m)&(plat==v)).nonzero().flatten(); ids=pid[avail].unique(); ids=ids[torch.randperm(len(ids))[:16]]
        if len(ids)<2: continue
        idx=torch.cat([avail[pid[avail]==y][torch.randint(0,(pid[avail]==y).sum(),(4,))] for y in ids]); words=[]; eot=[]
        for y in ids.tolist():
            for _ in range(4):
                text_ids=[SOT]+tok.encode('a photo of a X X X X X X X X person.')+[EOT]; eot.append(len(text_ids)-1); emb=base[text_ids].clone(); emb[5:9]=S[y]; emb[9:11]=U[m]; emb[11:13]=Q[v]; words.append(emb)
        out=F.normalize(text(torch.stack(words),torch.tensor(eot)),dim=1); logits=feat[idx]@out.t()/.07; target=torch.arange(len(ids)).repeat_interleave(4); loss=F.cross_entropy(logits,target); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([S,U,Q],1.); opt.step()
    bank=[]
    for y in range(n):
        for m,v in sorted(set(zip(mod[pid==y].tolist(),plat[pid==y].tolist()))):
            ids=[SOT]+tok.encode('a photo of a X X X X X X X X person.')+[EOT]; emb=base[ids].clone(); emb[5:9]=S[y]; emb[9:11]=U[m]; emb[11:13]=Q[v]; bank.append((text(emb[None],torch.tensor([len(ids)-1]))[0],y,m,v))
    torch.save({'text':F.normalize(torch.stack([x[0] for x in bank]),dim=1),'pid':torch.tensor([x[1] for x in bank]),'modality':torch.tensor([x[2] for x in bank]),'platform':torch.tensor([x[3] for x in bank]),'steps':args.steps},args.output)

if __name__=='__main__': main()
