"""Train M2-5 condition teacher for exactly 60 epochs on train-only cache."""
import argparse
from pathlib import Path
import torch
from model.m2_teacher import ConditionalFeatureTeacher,normalized_clip_logits

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--text-bank',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--lr',type=float,default=3.5e-4);p.add_argument('--device',default='cuda');a=p.parse_args();d=torch.device(a.device if torch.cuda.is_available() else 'cpu');c=torch.load(a.cache,map_location='cpu');tb=torch.load(a.text_bank,map_location='cpu');x=c['features'].float().to(d);y=c['pids'].long().to(d);m=c['modalities'].long().to(d);v=c['platforms'].long().to(d);bank=tb['text_bank'].float().to(d);teacher=ConditionalFeatureTeacher(dim=x.shape[1],modalities=3,platforms=2,rank=8,alpha=8).to(d);teacher.classifier=torch.nn.Linear(x.shape[1],int(y.max())+1).to(d);opt=torch.optim.Adam(teacher.parameters(),lr=a.lr,weight_decay=1e-4)
 for ep in range(a.epochs):
  opt.zero_grad();h=teacher(x,m,v);ce=torch.nn.functional.cross_entropy(teacher.classifier(h),y);log=normalized_clip_logits(h,torch.eye(h.shape[1],512,device=d),bank) if h.shape[1]==512 else h[:,:512]@bank.t()/0.07;txt=torch.nn.functional.cross_entropy(log,y);(ce+0.5*txt).backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0);opt.step()
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':teacher.state_dict(),'epochs':a.epochs,'temperature':.07,'text_bank':str(Path(a.text_bank).resolve())},a.out);print('teacher_epochs=%d'%a.epochs)
if __name__=='__main__':main()
