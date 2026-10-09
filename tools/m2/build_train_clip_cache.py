"""Validate and package a train-only CLIP feature cache for M2-1."""
import argparse, torch
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args(); x=torch.load(a.input,map_location='cpu',weights_only=False)
 for k in ('feature','pid','modality','platform'):
  if k not in x: raise ValueError('cache missing '+k)
 if len(x['feature'])!=len(x['pid']) or len(x['pid'])!=len(x['modality']): raise ValueError('cache lengths differ')
 if x['pid'].min()<0 or x['modality'].min()<0 or x['modality'].max()>2: raise ValueError('invalid train mapping')
 torch.save({k:x[k].cpu() if hasattr(x[k],'cpu') else x[k] for k in ('feature','pid','modality','platform')},a.output)
if __name__=='__main__': main()
