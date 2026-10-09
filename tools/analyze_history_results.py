import json,re,statistics,sys,argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser(description='解析 HI0-HI3 及历史 C0/T1/T2 结果，不训练模型。')
p.add_argument('--runs-root',default='E:/CVPR2027/ref/Trajectory_runs')
p.add_argument('--output',required=True)
args=p.parse_args()
ROOT=Path(args.runs_root);OUT=Path(args.output)
OUT.mkdir(exist_ok=True,parents=True)
PAT=r'Validation Results(?: - Epoch: (\d+))?[^\n]*\n[^\n]*INFO: mAP: ([\d.]+)%\n[^\n]*Rank-1\s*:\s*([\d.]+)%\n[^\n]*Rank-5\s*:\s*([\d.]+)%\n[^\n]*Rank-10\s*:\s*([\d.]+)%'
def evaluation(p):
 t=p.read_text(encoding='utf-8-sig'); matches=list(re.finditer(PAT,t)); m=matches[-1]
 return dict(metrics=list(map(float,m.groups()[1:])),line=t[:m.start()].count('\n')+1,path=str(p),pairs={f'{a}-{b}':list(map(float,v)) for a,b,*v in re.findall(r'Pair \(Q:(\w+), G:(\w+)\) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-5: ([\d.]+)%, Rank-10: ([\d.]+)%, mINP: ([\d.]+)%',t)},views={f'{a}-{b}':list(map(float,v)) for a,b,*v in re.findall(r'View \(Q:(\w+), G:(\w+)\) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-5: ([\d.]+)%, Rank-10: ([\d.]+)%, mINP: ([\d.]+)%',t)},weight=re.findall(r'  WEIGHT: (.*)',t),neck=re.findall(r'  NECK_FEAT: (.*)',t))
data={}
for name in ['HI0','HI1','HI2','HI3','C0','T1','T2']:
 d=ROOT/('history_innovation_1009' if name.startswith('HI') else 'trajectory_extensions_1002')/(name+'_seed1234')
 r={read:evaluation(next((d/f'eval_{read}BN').glob('test_log*.txt'))) for read in ['post','pre']}
 if name.startswith('HI'):
  p=next(d.glob('train_log*.txt'));t=p.read_text(encoding='utf-8-sig');c=(d/'console.log').read_text(encoding='utf-8-sig')
  rows=[json.loads(l) for l in (d/'history_epoch.jsonl').read_text().splitlines()]
  inline=[json.loads(l) for l in re.findall(r'History epoch summary: (\{.*\})',t)]
  r.update(config=yaml.safe_load((d/'resolved_config.yml').read_text()),source=(d/'source_commit.txt').read_text().strip(),status=(d/'source_status.txt').read_text().strip(),epochs=rows,inline_equal=inline==rows,curve={int(e):list(map(float,vals)) for e,*vals in re.findall(PAT,t)},runtime=re.findall(r'Runtime: (.*)',t),times=re.findall(r'Epoch (\d+) done\. Time per batch: ([\d.]+)\[s\] Speed: ([\d.]+)',t),epoch_done=list(map(int,re.findall(r'Epoch (\d+) done\.',t))),warnings=[l for l in c.splitlines() if re.search(r'Traceback|Error|Warning|\bnan\b|\binf\b',l,re.I)],train_path=str(p),console_evidence=[l for l in c.splitlines() if re.search(r'load|history|sample|train|query|gallery',l,re.I)][:65])
 data[name]=r
(OUT/'parsed.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
for n,r in data.items():
 print(n, 'post',r['post']['metrics'],'pre',r['pre']['metrics'])
 if not n.startswith('HI'):continue
 print('source',r['source'],r['status'],'epochs',len(r['epochs']),'inline',r['inline_equal'],'times',len(r['times']),'skips',sum(x['amp_skipped_steps'] for x in r['epochs']))
 for e in [1,2,5,10,20,30,40,50,60]:
  x=r['epochs'][e-1];L=x['layers'];g=x['gain_distribution'];print(e,'ce/tri/pred',*[round(x[k],5) for k in ['ce','triplet','prediction_loss']],'gain',g['patch']['rms'],'maxR',max(v['correction_ratio'] for v in L.values()),'maxCLS',max(v.get('cls_correction_ratio',0) for v in L.values()),'PG',round(statistics.mean(v['prediction_gain_ratio_of_means'] for v in L.values() if 'prediction_gain_ratio_of_means' in v),4),'clsGain',g.get('cls',{}).get('rms'))
 print('curve',r['curve'])
