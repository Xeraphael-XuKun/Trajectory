import json,re,statistics,argparse
from pathlib import Path
import yaml

p=argparse.ArgumentParser(description='读取HD四组诊断与原HI3前5epoch，核对并提取分析数据。')
p.add_argument('--runs-root',default='E:/CVPR2027/ref/Trajectory_runs')
p.add_argument('--output',required=True)
args=p.parse_args();root=Path(args.runs_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
def read(p):return p.read_text(encoding='utf-8-sig')
def lines(p):return [json.loads(x) for x in read(p).splitlines()]
def flat(x,p=''):
 result={}
 for k,v in x.items():
  if isinstance(v,dict):result.update(flat(v,p+k+'.'))
  else:result[p+k]=v
 return result
data={}
for i in range(4):
 name='HD'+str(i);directory=root/'history_optimization_1009'/(name+'_e5_seed1234')
 tpath=next(directory.glob('train_log_*.txt'));t=read(tpath);console=read(directory/'console.log')
 es=lines(directory/'history_epoch.jsonl');gs=lines(directory/'history_gradients.jsonl')
 c=yaml.safe_load(read(directory/'resolved_config.yml'));initial=json.loads(read(directory/'history_optimizer_initial.json'))
 stop=json.loads(read(directory/'history_stop.json'))
 r=dict(config=c,initial=initial,stop=stop,epochs=es,gradients=gs,source=read(directory/'source_commit.txt').strip(),status=read(directory/'source_status.txt').strip(),runtime=re.findall(r'Runtime: (.*)',t),train_path=str(tpath),inline_equal=[json.loads(s) for s in re.findall(r'History epoch summary: (\{.*\})',t)]==es,
        epoch_done=list(map(int,re.findall(r'Epoch (\d+) done\.',t))),times=re.findall(r'Epoch (\d+) done\. Time per batch: ([\d.]+)\[s\]',t),checkpoints=[str(x) for x in directory.glob('*.pth')],fatal_lines=[s for s in console.splitlines() if re.search(r'Traceback|OutOfMemoryError|\bLoss: (nan|inf)\b',s,re.I)])
 r['checks']={'epochs_1_to_5':[e['epoch'] for e in es]==list(range(1,6)),
              'stop_and_horizon':stop['completed_epoch']==5 and stop['schedule_epochs']==60 and c['SOLVER']['MAX_EPOCHS']==60 and c['HISTORY']['STOP_AFTER_EPOCH']==5,
              'gradient_keys_complete':[(g['epoch'],g['iteration']) for g in gs]==[(e['epoch'],j) for e in es for j in range(1,e['iterations']+1) if j==1 or j%50==0],
              'lr_changes_over_epochs':len(set(e['optimization']['parameters']['gain']['learning_rates'][0] for e in es))>1}
 r['gradient_epoch_summary']={}
 for e in es:
  good=[g for g in gs if g['epoch']==e['epoch'] and not g['amp_skipped']]
  summary={'valid':len(good),'sampled':sum(g['epoch']==e['epoch'] for g in gs),
           'global_grad_median':statistics.median(g['global_grad_preclip_norm'] for g in good), 'families':{}}
  for f in ['gain','anchors','correctors','predictors']:
   fields={}
   for field in ['identity_grad_norm','prediction_raw_grad_norm','data_grad_postclip_norm','decay_term_norm','decay_to_data_norm_ratio']:
    v=[g[field][f] for g in good if g[field][f] is not None]
    fields[field]={'median':statistics.median(v) if v else None,'min':min(v) if v else None,'max':max(v) if v else None,'n':len(v),'n_gt_1':sum(x>1 for x in v)}
   summary['families'][f]=fields
  summary['predictor_aux_to_identity_median']=statistics.median(g['prediction_weighted_grad_norm']['predictors']/g['identity_grad_norm']['predictors'] for g in good if g['identity_grad_norm']['predictors'] is not None and g['identity_grad_norm']['predictors']>0)
  r['gradient_epoch_summary'][e['epoch']]=summary
 data[name]=r
 print(name,'source',r['source'],'status',r['status'],'checks',r['checks'],'inline',r['inline_equal'],'ckpts',len(r['checkpoints']),'steps',sum(e['iterations'] for e in es),'skips',sum(e['amp_skipped_steps'] for e in es),'grad records',len(gs),'grad skipped',sum(g['amp_skipped'] for g in gs))
 for e in es:
  ps=e['optimization']['parameters'];cr=[v['correction_ratio'] for v in e['layers'].values()];ep=e['epoch'];summary=r['gradient_epoch_summary'][ep]
  print(ep,'ce tri pred',*[round(e[k],6) for k in ['ce','triplet','prediction_loss']], 'effgain',ps['gain']['effective_rms'],'anchor/corrector',ps['anchors']['rms'],ps['correctors']['rms'],'R mean/max',statistics.mean(cr),max(cr),'last',cr[-1], 'clipNormMed',summary['global_grad_median'],'WD/data',[summary['families'][f]['decay_to_data_norm_ratio']['median'] for f in ['gain','anchors','correctors','predictors']],'aux/id',summary['predictor_aux_to_identity_median'])
 print('R layers e5',[round(v['correction_ratio']*100,4) for v in es[-1]['layers'].values()])
 print('PG e5',statistics.mean(v['prediction_gain_ratio_of_means'] for v in es[-1]['layers'].values() if 'prediction_gain_ratio_of_means' in v))
for n in ['HD1','HD2','HD3']:
 ref=flat(data['HD0']['config']);c=flat(data[n]['config']);print(n,'diff',{k:v for k,v in c.items() if v!=ref[k]})
hi3=lines(root/'history_innovation_1009/HI3_seed1234/history_epoch.jsonl')[:5]
differences=[]
for a,b in zip(hi3,data['HD0']['epochs']):
 # 新增optimization诊断不参与原有字段一致性比较。
 for key,value in flat(a).items():
  actual=flat(b)[key]
  if actual!=value:differences.append({'epoch':a['epoch'],'field':key,'old':value,'new':actual})
data['reference']={'HI3_first5_exact_equal_to_HD0':not differences,'difference_count':len(differences),'differences':differences}
print('HI3 vs HD0',data['reference']['HI3_first5_exact_equal_to_HD0'],'diffcount',len(differences),differences[:3])
(out/'parsed_hd.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
