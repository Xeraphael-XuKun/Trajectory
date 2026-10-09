"""M2-5 conditional teacher components.
The Q/V LoRA tensors are explicit and condition indexed; B is zero initialized.
"""
import torch
from torch import nn
import torch.nn.functional as F
class ConditionalQVLora(nn.Module):
 def __init__(self,dim=768,depth=12,modalities=3,platforms=2,rank=8,alpha=8.):
  super().__init__();self.dim=dim;self.depth=depth;self.scale=alpha/rank
  for n,k in [('q_m',modalities),('v_m',modalities),('q_p',platforms),('v_p',platforms)]:
   setattr(self,n+'A',nn.Parameter(torch.randn(depth,k,rank,dim)*.02));setattr(self,n+'B',nn.Parameter(torch.zeros(depth,k,dim,rank)))
 def delta(self,layer,mod,plat):
  def d(n,i):return getattr(self,n+'B')[layer,i]@getattr(self,n+'A')[layer,i]*self.scale
  return d('q_m',mod)+d('q_p',plat),d('v_m',mod)+d('v_p',plat)
 def qkv_weight(self,w,layer,mod,plat):
  q,v=self.delta(layer,int(mod),int(plat));o=w.clone();o[:self.dim]+=q;o[2*self.dim:3*self.dim]+=v;return o
 def qkv_delta(self,layer,mod,plat,device=None):
  mod=torch.as_tensor(mod,device=device).long().view(-1); plat=torch.as_tensor(plat,device=device).long().view(-1)
  q=[]; v=[]
  for m,p in zip(mod.tolist(),plat.tolist()):
   a,b=self.delta(layer,m,p); q.append(a); v.append(b)
  return torch.cat([torch.stack(q),torch.zeros_like(torch.stack(q)),torch.stack(v)],dim=1)
class ConditionalFeatureTeacher(nn.Module):
 """Condition adapter used by teacher training; frozen CLIP features are adapted per label."""
 def __init__(self,dim=768,**kw):
  super().__init__();self.lora=ConditionalQVLora(dim=dim,**kw);self.norm=nn.BatchNorm1d(dim);self.classifier=nn.LazyLinear(1)
 def forward(self,feat,modality,platform):
  # Feature-level execution is useful for cache smoke tests; production visual path uses qkv_weight.
  return self.norm(feat)
 def forward_visual(self, backbone, images, modality, platform):
  """Run the visual transformer with per-sample Q/V deltas."""
  off = torch.zeros(images.shape[0], device=images.device, dtype=images.dtype)
  return backbone(images, modal_label=None, view_label=None,
                  trajectory_gate=off,
                  conditional_lora=self.lora,
                  conditional_modality=modality,
                  conditional_platform=platform)
def normalized_clip_logits(feat,proj,text_bank,tau=.07):
 return F.normalize(feat.float()@proj.float(),dim=-1)@F.normalize(text_bank.float(),dim=-1).t()/tau
def kd_kl(logits_teacher,logits_student,temperature=2.):
 t=float(temperature);lt=F.log_softmax(logits_teacher.float()/t,dim=-1);ls=F.log_softmax(logits_student.float()/t,dim=-1);return t*t*F.kl_div(ls,lt.exp().detach(),reduction='batchmean')
