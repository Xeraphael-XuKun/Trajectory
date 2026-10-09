import torch
import torch.nn.functional as F

def load_text_bank(path, device=None):
    bank=torch.load(path,map_location=device or 'cpu',weights_only=False)
    if not isinstance(bank,dict) or not {'text','pid','modality'}.issubset(bank):
        raise ValueError('M2-1 text bank must contain text, pid and modality')
    return bank

def _group(scores,pids,ids):
    return torch.stack([torch.logsumexp(scores[:,pids==y],1)-torch.log((pids==y).sum().float()) for y in ids],1)

def condition_bridge_loss(z, labels, modalities, bank, temperature=.07, all_weight=.5, cross_weight=.5):
    text=F.normalize(bank['text'].float().to(z.device),dim=1); pids=torch.as_tensor(bank['pid'],device=z.device).long(); mods=torch.as_tensor(bank['modality'],device=z.device).long(); z=F.normalize(z.float(),dim=1)
    ids=torch.arange(int(pids.max())+1,device=z.device); scores=z@text.t()/temperature
    la=F.cross_entropy(_group(scores,pids,ids),labels.long()); vals=[]
    for i,m in enumerate(torch.as_tensor(modalities,device=z.device).tolist()):
        keep=mods!=int(m); cand=pids[keep].unique()
        if cand.numel()<2 or labels[i].item() not in cand.tolist(): continue
        q=scores[i,keep]; pp=pids[keep]; g=torch.stack([torch.logsumexp(q[pp==y],0)-torch.log((pp==y).sum().float()) for y in cand.tolist()])
        vals.append(F.cross_entropy(g[None],torch.tensor([cand.tolist().index(int(labels[i]))],device=z.device)))
    lx=torch.stack(vals).mean() if vals else z.new_zeros(())
    return all_weight*la+cross_weight*lx, {'all':la.detach(),'cross':lx.detach(),'valid':len(vals)}
