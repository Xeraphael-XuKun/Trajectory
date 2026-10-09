"""Build M2-4 relation bank from a previously audited RGB-center tensor."""
import argparse, torch
from m2_cache import build_relation_cache
p=argparse.ArgumentParser(); p.add_argument('--centers',required=True); p.add_argument('--pid-order',required=True, help='torch file containing 1D train PID order'); p.add_argument('--output',required=True); a=p.parse_args()
c=torch.load(a.centers,map_location='cpu',weights_only=False); ids=torch.load(a.pid_order,map_location='cpu',weights_only=False)
meta={}
if isinstance(c,dict):
    meta.update(c.get('metadata',{})); c=c.get('centers',c.get('features'))
if isinstance(ids,dict): ids=ids.get('pid_order',ids.get('pids'))
build_relation_cache(c,ids,a.output,metadata=meta)
print('saved',a.output)
