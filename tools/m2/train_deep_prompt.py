"""M2-3 stage A: train identity tokens for 10,000 real optimizer steps."""
import argparse, torch
from torch.nn import functional as F
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer, SOT, EOT
from model.make_model import _read_clip_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',required=True); ap.add_argument('--clip',required=True); ap.add_argument('--output',required=True); ap.add_argument('--steps',type=int,default=10000); a=ap.parse_args()
    c=torch.load(a.cache,map_location='cpu',weights_only=False); z=F.normalize(c['feature'].float(),dim=1); pid=torch.as_tensor(c['pid']); n=int(pid.max())+1
    text=CLIPTextEncoder(); text.load_clip(_read_clip_checkpoint(a.clip)); text.freeze(); tok=build_tokenizer(); base=text.token_embedding.weight.detach()
    S=torch.nn.Parameter(torch.randn(n,4,512)*.02); opt=torch.optim.Adam([S],lr=3.5e-4,weight_decay=1e-4)
    ids_template=[SOT]+tok.encode('a photo of a X X X X person .')+[EOT]; eot=len(ids_template)-1
    for _ in range(a.steps):
        ids=pid.unique(); ids=ids[torch.randperm(len(ids))[:16]]
        if len(ids)<2: continue
        rows=torch.cat([torch.nonzero(pid==y).flatten()[torch.randint(0,int((pid==y).sum()),(4,))] for y in ids])
        emb=base[ids_template].unsqueeze(0).expand(len(ids),-1,-1).clone(); emb[:,5:9]=S[ids]
        t=F.normalize(text(emb,torch.full((len(ids),),eot,dtype=torch.long)),dim=-1); logits=z[rows]@t.t()/.07
        target=torch.arange(len(ids)).repeat_interleave(4); loss=F.cross_entropy(logits,target); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([S],1.); opt.step()
    with torch.no_grad():
        emb=base[ids_template].unsqueeze(0).expand(n,-1,-1).clone(); emb[:,5:9]=S; anchors=F.normalize(text(emb,torch.full((n,),eot,dtype=torch.long)),dim=-1)
    torch.save({'id_tokens':S.detach().cpu(),'anchors':anchors.cpu(),'steps':a.steps,'template':'D slots before four ID slots'},a.output)
if __name__=='__main__': main()
