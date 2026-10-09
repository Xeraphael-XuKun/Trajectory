"""HD核心隔离/诊断检查；可选真实GPU短程测试，不代表正式5epoch或60epoch结果。"""
import argparse
import copy
import gc
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import cfg
from model.backbones.history_adapter import HistoryInnovationAdapter
from loss.history_prediction import history_prediction_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.history_optimization import HistoryOptimizationMonitor, diagnostic_stop_epoch


class Toy(nn.Module):
    def __init__(self,scope='patch'):
        super().__init__(); self.base=nn.Module()
        self.base.stem=nn.Linear(16,16)
        self.base.history_adapter=HistoryInnovationAdapter(16,3,4,rank=4,alpha=.25,gain_mode='linear',token_scope=scope)
        self.base.blocks=nn.ModuleList([nn.Linear(16,16) for _ in range(3)])
        self.classifier=nn.Linear(16,2)
    def forward(self,x):
        x=self.base.stem(x);history=[];previous=None;items={}
        for i,b in enumerate(self.base.blocks):
            r,history,previous,item=self.base.history_adapter.correction(x,i,history,previous,True,True)
            items[i+1]=item;x=b(x+r)
        identity=self.classifier(x.mean(1)).square().mean()
        prediction=history_prediction_loss({'layers':items,'prediction_layers':[2,3]})
        return identity,prediction


def configuration(name):
    c=cfg.clone();c.merge_from_file(str(ROOT/'configs'/('history_'+name+'.yml')));return c


def run_cpu(output):
    torch.set_num_threads(4);report={}
    configs={n:configuration(n) for n in ['HI0','HI1','HI2','HI3','HD0','HD1','HD2','HD3']}
    original={}
    source=subprocess.check_output(['git','-C',str(ROOT),'show','ff5efd4:solver/make_optimizer.py']).decode('utf-8')
    exec(compile(source,'reference_optimizer','exec'),original)
    torch.manual_seed(1234);model=Toy(scope='all');center=nn.Linear(2,2)
    for n in ['HI0','HI1','HI2','HI3']:
        c=configs[n];a,_=make_optimizer(c,model,center);b,_=original['make_optimizer'](c,model,center)
        assert [(g['lr'],g['weight_decay']) for g in a.param_groups]==[(g['lr'],g['weight_decay']) for g in b.param_groups]
    report['HI0_HI3_original_optimizer_unchanged']=True
    maps={}
    for n in ['HD0','HD1','HD2','HD3']:
        c=configs[n];opt,_=make_optimizer(c,model,center)
        maps[n]={key:{'lr':g['lr'],'weight_decay':g['weight_decay']} for (key,p),g in zip(model.named_parameters(),opt.param_groups)}
        for key,values in maps[n].items():
            family=key.split('.')[2] if key.startswith('base.history_adapter.') else ''
            expected=0. if n in ('HD2','HD3') and family in ('gain','cls_gain','anchors','correctors') else 1e-4
            assert values['weight_decay']==expected,(n,key,values)
    report['optimizer_groups']=maps
    for name in ['HD1','HD2','HD3']:
        adjusted=configs[name].clone()
        for key in ['ALPHA','WRITE_WEIGHT_DECAY']:
            setattr(adjusted.HISTORY,key,getattr(configs['HD0'].HISTORY,key))
        adjusted.OUTPUT_DIR=configs['HD0'].OUTPUT_DIR
        assert adjusted.dump()==configs['HD0'].dump()
    report['HD_matrix_only_two_method_factors']=True
    hi3=configs['HI3'].clone();hi3.HISTORY.WRITE_WEIGHT_DECAY=0;hi3.HISTORY.ENABLED=False
    a,_=make_optimizer(hi3,model,center);b,_=original['make_optimizer'](hi3,model,center)
    assert [(g['lr'],g['weight_decay']) for g in a.param_groups]==[(g['lr'],g['weight_decay']) for g in b.param_groups]
    report['disabled_history_no_optimizer_override']=True
    # 日程不因STOP=5缩短；不是改MAX_EPOCHS=5的短周期。
    a,_=make_optimizer(configs['HI3'],model,center);b,_=make_optimizer(configs['HD0'],model,center)
    sa=create_scheduler(configs['HI3'],a,iters_per_epoch=469);sb=create_scheduler(configs['HD0'],b,iters_per_epoch=469)
    for step in [0,1,99,100,468,469,2344,4689,28139]:
        sa.step_update(step);sb.step_update(step)
        assert [g['lr'] for g in a.param_groups]==[g['lr'] for g in b.param_groups]
    assert diagnostic_stop_epoch(configs['HD0'])==5 and configs['HD0'].SOLVER.MAX_EPOCHS==60
    report['60_epoch_schedule_preserved']=True
    # 相同图上测量额外梯度，正式.grad、RNG与Adam更新不变；零gain/非零gain都覆盖。
    reference=Toy();measured=copy.deepcopy(reference)
    o1,_=make_optimizer(configs['HD0'],reference,center);o2,_=make_optimizer(configs['HD0'],measured,center)
    mon=HistoryOptimizationMonitor(measured.base.history_adapter,o2)
    x=torch.randn(6,5,16)
    for step in range(3):
        o1.zero_grad();o2.zero_grad()
        a,b=reference(x);(a+b).backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1);o1.step()
        a,b=measured(x);rng=torch.random.get_rng_state().clone()
        prior=[None if p.grad is None else p.grad.clone() for p in measured.parameters()]
        stats=mon.loss_gradients(a,b,1.,128.)
        assert torch.equal(rng,torch.random.get_rng_state())
        assert all((p.grad is None if g is None else torch.equal(p.grad,g)) for p,g in zip(measured.parameters(),prior))
        assert stats['prediction_raw_grad_norm']['gain']==0 and stats['prediction_raw_grad_norm']['anchors']==0
        (a+b).backward();mon.before_clip();torch.nn.utils.clip_grad_norm_(measured.parameters(),1)
        stats.update(mon.after_clip());json.dumps(stats,allow_nan=False);o2.step()
        assert all(torch.equal(p,q) for p,q in zip(reference.parameters(),measured.parameters()))
    report['diagnostic_does_not_change_rng_grad_or_updates_CPU']=True
    # alpha改变只是公式缩放；零gain基线初始化保持不变。
    h=torch.randn(2,5,16);adapter=measured.base.history_adapter
    adapter.gain.data.fill_(.01)
    adapter.alpha=.25;r1=adapter.correction(h,0,[],None)[0]
    adapter.alpha=1.;r2=adapter.correction(h,0,[],None)[0]
    assert torch.equal(r2,r1*4)
    report['alpha_scaling_exact']=True
    # HF生成器只在显式选O后产出配置；三个正式组的差异必须精准。
    spec=importlib.util.spec_from_file_location('prepare',ROOT/'tools/prepare_history_followup.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for choice in ['HD0','HD1','HD2','HD3']:
        mod.prepare(choice,output/'followup_configs')
        parsed=[]
        for n in ['HF1','HF2','HF3']:
            c=cfg.clone();c.merge_from_file(str(output/'followup_configs'/choice/('history_'+n+'.yml')))
            assert c.HISTORY.STOP_AFTER_EPOCH==0 and c.SOLVER.MAX_EPOCHS==60
            assert c.HISTORY.ALPHA==configs[choice].HISTORY.ALPHA and c.HISTORY.WRITE_WEIGHT_DECAY==configs[choice].HISTORY.WRITE_WEIGHT_DECAY
            parsed.append(c)
        assert [(c.HISTORY.TOKEN_SCOPE,c.HISTORY.LOSS_WEIGHT) for c in parsed]==[('patch',1.),('patch',0.),('all',1.)]
    report['all_HF_choices_valid']=True
    (output/'cpu_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HD_CPU_CHECKS_OK')


def run_gpu(args,output):
    from train import set_seed
    from datasets import make_dataloader
    from model import make_model
    from loss import make_loss
    from processor import do_train
    from utils.logger import setup_logger
    from utils.runtime import runtime_summary
    class Short:
        def __init__(self,batch):self.batch=batch;self.batch_size=len(batch[1])
        def __len__(self):return 12
        def __iter__(self):yield from [self.batch]*12
    results=[];first_state=None;first_epochs=None
    for period in [0,4]:
        dest=output/('gpu_diag_'+str(period));dest.mkdir()
        c=configuration('HD0');c.HISTORY.STOP_AFTER_EPOCH=2;c.HISTORY.GRAD_DIAG_PERIOD=period
        c.MODEL.PRETRAIN_PATH=args.pretrained;c.DATASETS.ROOT_DIR=args.data_root
        c.DATALOADER.NUM_WORKERS=0;c.DATALOADER.NUM_INSTANCE=2;c.SOLVER.IMS_PER_BATCH=4
        c.OUTPUT_DIR=str(dest);c.freeze();set_seed(1234,True)
        setup_logger('transreid',str(dest),if_train=True)
        (dest/'resolved_config.yml').write_text(c.dump(),encoding='utf-8')
        loader,_,_,_,classes,cameras,views=make_dataloader(c);batch=next(iter(loader));short=Short(batch)
        model=make_model(c,classes,cameras,views);loss,center=make_loss(c,classes)
        opt,co=make_optimizer(c,model,center);sched=create_scheduler(c,opt,iters_per_epoch=len(short))
        do_train(c,model,center,short,{},opt,co,sched,loss,{},0)
        epochs=[json.loads(l) for l in (dest/'history_epoch.jsonl').read_text().splitlines()]
        assert [e['epoch'] for e in epochs]==[1,2]
        assert (dest/'transformer_2.pth').exists() and not (dest/'transformer_60.pth').exists()
        stop=json.loads((dest/'history_stop.json').read_text());assert stop['schedule_epochs']==60
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if first_state is None:first_state=state;first_epochs=epochs
        else:
            assert all(torch.equal(first_state[k],v) for k,v in state.items()),'GPU diagnostics changed parameters'
            grads=[json.loads(l) for l in (dest/'history_gradients.jsonl').read_text().splitlines()]
            assert len(grads)==8
            assert sum(e['amp_skipped_steps'] for e in epochs)==sum(e['amp_skipped_steps'] for e in first_epochs)
        results.append({'diag_period':period,'attempts':24,'amp_skipped':sum(e['amp_skipped_steps'] for e in epochs)})
        del model,opt,co,sched,loss,center,loader,batch,short;gc.collect();torch.cuda.empty_cache()
    (output/'gpu_smoke.json').write_text(json.dumps({'runtime':runtime_summary(),'runs':results,'on_off_weights_exact_equal':True,'formal_training':False},indent=2),encoding='utf-8')
    print('HD_GPU_SMOKE_OK')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--pretrained',default='');p.add_argument('--data-root',default='')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    run_cpu(out)
    if args.gpu_smoke:run_gpu(args,out)
