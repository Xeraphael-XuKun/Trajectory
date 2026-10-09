"""M2-5 stage A: 10,000 real optimizer updates of shared identity tokens."""
import argparse, math
from pathlib import Path
import torch
from model.backbones.clip_text import CLIPTextEncoder,build_tokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--clip',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=10000);p.add_argument('--device',default='cuda');a=p.parse_args();d=torch.device(a.device if torch.cuda.is_available() else 'cpu');c=torch.load(a.cache,map_location='cpu');z=c['features'].float();pid=c['pids'].long();Y=int(pid.max())+1;sd=torch.load(a.clip,map_location='cpu');te=CLIPTextEncoder().to(d);te.load_clip(sd);te.eval();tok=build_tokenizer();base=[49406]+tok.encode('a photo of a')+[0,0,0,0]+tok.encode('person.')+[49407];slots=[5,6,7,8];ids=torch.tensor(base+[0]*(77-len(base)),device=d);tokens=torch.nn.Parameter(torch.randn(Y,4,512,device=d)*.02);opt=torch.optim.Adam([tokens],lr=3.5e-4,weight_decay=1e-4);z=z.to(d);pid=pid.to(d);g=torch.Generator(device=d).manual_seed(1234)
 for step in range(a.steps):
  chosen=torch.randint(len(z),(min(64,len(z)),),generator=g,device=d); yy=pid[chosen]; emb=te.token_embedding(ids.unsqueeze(0).expand(Y,-1)).clone(); emb[:,slots]=tokens; text=torch.nn.functional.normalize(te(emb,torch.full((Y,),len(base)-1,device=d)).float(),dim=-1); logits=z[chosen]@text.t()/0.07;loss=torch.nn.functional.cross_entropy(logits,yy);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([tokens],1.0);opt.step()
  if step+1==a.steps: print('prompt_updates=%d loss=%.6f'%(step+1,loss.item()))
 emb=te.token_embedding(ids.unsqueeze(0).expand(Y,-1)).clone();emb[:,slots]=tokens;bank=torch.nn.functional.normalize(te(emb,torch.full((Y,),len(base)-1,device=d)).float(),dim=-1).cpu();Path(a.out).parent.mkdir(parents=True,exist_ok=True);torch.save({'tokens':tokens.detach().cpu(),'text_bank':bank,'pid_map':c.get('pid_map',{}),'steps':a.steps,'temperature':.07},a.out)
if __name__=='__main__':main()
