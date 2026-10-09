"""M2-5 P-ID: condition-balanced, bidirectional multi-positive text learning."""
import argparse
import math
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer
from tools.m2.artifacts import read_clip, file_digest, provenance, rng_state

def prompt_loss(z, text, row_pids, text_pids, temperature=.07):
    logits = z.float() @ text.float().t() / temperature
    target = (row_pids[:, None] == text_pids[None, :]).long().argmax(1)
    image_loss = F.cross_entropy(logits, target)
    log_prob = logits.t().log_softmax(dim=1)
    positives = text_pids[:, None] == row_pids[None, :]
    text_loss = -((log_prob * positives).sum(1) / positives.sum(1)).mean()
    return .5 * (image_loss + text_loss)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--clip',required=True); p.add_argument('--out',required=True); p.add_argument('--steps',type=int,default=10000); p.add_argument('--device',default='cuda'); a=p.parse_args()
    torch.manual_seed(1234); random.seed(1234)
    dev=torch.device(a.device)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    if cache.get('split') != 'train': raise ValueError('P-ID requires a train-only cache')
    z=cache['clip_features'].float(); pids=cache['pids'].long(); mods=cache['modalities'].long(); plats=cache['platforms'].long()
    Y=len(cache['pid_map']); groups={}; skipped=[]
    for condition in sorted(set(zip(mods.tolist(),plats.tolist()))):
        mask=(mods==condition[0]) & (plats==condition[1]); members={int(y):torch.where(mask & (pids==y))[0] for y in pids[mask].unique()}
        if len(members)<2: skipped.append(condition)
        else: groups[condition]=members
    if not groups: raise ValueError('no train condition contains two identities')
    te=CLIPTextEncoder().to(dev); te.load_clip(read_clip(a.clip)); te.eval()
    tok=build_tokenizer(); sentence='a photo of a X X X X person.'; tokens_sentence=[49406]+tok.encode(sentence)+[49407]; placeholder=tok.encode('X')
    if len(placeholder)!=1: raise ValueError('X must tokenize to a single token')
    slots=[i for i,t in enumerate(tokens_sentence) if t==placeholder[0]]
    if len(slots)!=4: raise ValueError('P-ID requires four actual token positions')
    ids=torch.tensor(tokens_sentence+[0]*(77-len(tokens_sentence)),device=dev); eot=int(torch.where(ids==49407)[0].item())
    context=torch.nn.Parameter(torch.randn(Y,4,512,device=dev)*.02); opt=torch.optim.Adam([context],lr=3.5e-4,weight_decay=1e-4)
    def encode(rows):
        embeddings=te.token_embedding(ids[None].expand(len(rows),-1)).clone(); embeddings[:,slots]=context[rows]
        return F.normalize(te(embeddings,torch.full((len(rows),),eot,device=dev,dtype=torch.long)).float(),dim=-1)
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step: min(1.,(step+1)/100.) if step<100 else .5*(1+math.cos(math.pi*(step-100)/max(1,a.steps-100))))
    conditions=[]; start=time.time(); counts={str(k):0 for k in groups}
    for step in range(a.steps):
        if not conditions: conditions=list(groups); random.shuffle(conditions)
        condition=conditions.pop(); members=groups[condition]; selected=random.sample(sorted(members),min(16,len(members))); indices=[]
        for y in selected:
            available=members[y].tolist(); indices.extend(random.sample(available,4) if len(available)>=4 else random.choices(available,k=4))
        rows=torch.tensor(selected,device=dev); batch_pids=pids[indices].to(dev); text=encode(rows); loss=prompt_loss(z[indices].to(dev),text,batch_pids,rows)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([context],1.); opt.step(); schedule.step(); counts[str(condition)]+=1
        if (step+1)%100==0: print('prompt_step=%d loss=%.6f condition=%s'%(step+1,loss.item(),condition))
    with torch.no_grad(): text_bank=torch.cat([encode(torch.arange(i,min(i+32,Y),device=dev)) for i in range(0,Y,32)]).cpu()
    meta=provenance(cache,a.clip); meta.update({'template':sentence,'token_slots':slots,'eot_index':eot,'stage':'prompt','updates':a.steps,'cache_digest':file_digest(a.cache)})
    condition_mask=cache.get('condition_mask',{'modalities':mods,'platforms':plats})
    payload={'tokens':context.detach().cpu(),'text_bank':text_bank,'pid_map':cache['pid_map'],'condition_mask':condition_mask,'steps':a.steps,'temperature':.07,'metadata':meta,'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'rng':rng_state(),'budget':{'seconds':time.time()-start,'optimizer_steps':a.steps,'condition_steps':counts,'skipped_conditions':skipped}}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,a.out)
    torch.save({'text_bank':text_bank,'pid_map':cache['pid_map'],'condition_mask':condition_mask,'metadata':meta},str(Path(a.out).with_name('shared_id_text_bank.pt')))
    print('prompt_updates=%d output=%s'%(a.steps,a.out))
if __name__=='__main__': main()
