"""HI4 local fidelity checks: original linear HI3, changing only token scope."""
import argparse
import json
import random
import subprocess
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from config import cfg
from model import make_model
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer


def seed():
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)


def main(args):
    torch.set_num_threads(4)
    c3=cfg.clone();c3.merge_from_file(str(ROOT/'configs/history_HI3.yml'))
    c4=cfg.clone();c4.merge_from_file(str(ROOT/'configs/history_HI4.yml'))
    match=c3.clone();match.HISTORY.TOKEN_SCOPE='all';match.OUTPUT_DIR=c4.OUTPUT_DIR
    assert match.dump()==c4.dump(), 'HI4 must differ only in scope/output after merging defaults'
    assert c4.HISTORY.GAIN_MODE=='linear' and c4.HISTORY.ALPHA==0.25
    assert c4.HISTORY.WRITE_WEIGHT_DECAY==-1 and c4.HISTORY.STOP_AFTER_EPOCH==0 and c4.HISTORY.GRAD_DIAG_PERIOD==0
    # Ensure old forward/loss definitions have not been swapped for HF/P3/teacher variants.
    paths=['model/backbones/history_adapter.py','model/backbones/vit_pytorch.py','model/make_model.py','loss/history_prediction.py']
    subprocess.check_call(['git','-C',str(ROOT),'diff','--exit-code','b44e69c','HEAD','--']+paths)
    c4.MODEL.PRETRAIN_PATH=args.pretrained
    seed();model=make_model(c4,500,7,2)
    adapter=model.base.history_adapter
    assert adapter.token_scope=='all' and adapter.gain_mode=='linear'
    assert adapter.cls_gain is not None and model.base.token_trajectory is None
    assert sum(p.numel() for p in adapter.parameters())==2662528
    optimizer,_=make_optimizer(c4,model,nn.Linear(1,1))
    # Original b44e69c optimizer should produce identical lr/decay for every parameter.
    old=types.ModuleType('reference_optimizer')
    source=subprocess.check_output(['git','-C',str(ROOT),'show','b44e69c:solver/make_optimizer.py']).decode()
    exec(compile(source,'old_optimizer','exec'),old.__dict__)
    original,_=old.make_optimizer(c4,model,nn.Linear(1,1))
    assert [(id(g['params'][0]),g['lr'],g['weight_decay']) for g in optimizer.param_groups]==[(id(g['params'][0]),g['lr'],g['weight_decay']) for g in original.param_groups]
    del original
    if args.data_root:
        from datasets import make_dataloader
        d=c4.clone();d.DATASETS.ROOT_DIR=args.data_root;d.DATALOADER.NUM_WORKERS=0
        d.DATALOADER.NUM_INSTANCE=2;d.SOLVER.IMS_PER_BATCH=4
        loader,*_=make_dataloader(d);images,labels,_=next(iter(loader))
    else:
        images=[torch.randn(2,3,256,128) for _ in range(3)];labels=torch.tensor([0,1])
    target=labels.repeat(3)
    model.train();out=model(images,history_diagnostics=True)
    assert all(item['prediction'].shape[1]==128 for l,item in out[3]['layers'].items() if l>1)
    pred=history_prediction_loss(out[3]);task=nn.functional.cross_entropy(out[0],target)+TripletLoss()(out[1],target)[0]
    (task+pred).backward()
    assert adapter.cls_gain.grad.abs().sum()>0 and adapter.gain.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad()
    assert adapter.cls_gain.detach().abs().sum()>0
    with torch.no_grad():probe=model(images,history_diagnostics=True)
    assert any(item['diagnostics']['cls_correction_rms']>0 for item in probe[3]['layers'].values())
    del optimizer,out,probe
    model.eval()
    with torch.no_grad():after=model(images[0],mode=1)
    output=ROOT/'work/hi4_check';output.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),output/'check.pth')
    with torch.no_grad():adapter.cls_gain.add_(1)
    model.load_param(str(output/'check.pth'))
    with torch.no_grad():assert torch.equal(after,model(images[0],mode=1))
    # Same-size HI2 tanh config must not silently load a linear HI4 checkpoint.
    oldspec=adapter.specification.clone();adapter.specification[-1]=0
    try:
        model.load_param(str(output/'check.pth'))
    except ValueError:
        pass
    else:
        raise AssertionError('wrong gain specification accepted')
    adapter.specification.copy_(oldspec)
    report={'status':'LOCAL_CPU_CHECK_ONLY','token_scope':'all','gain_mode':'linear','alpha':0.25,
            'single_scope_difference_from_hi3':True,'original_forward_loss_source_unchanged':True,
            'original_optimizer_groups_equal':True,'prediction_loss_patch_only':True,
            'real_data':bool(args.data_root),'actual_images':sum(len(x) for x in images),
            'patch_and_cls_gain_updated':True,'checkpoint_reload_equal':True,'wrong_gain_config_rejected':True,
            'adapter_parameters':2662528,'torch':torch.__version__}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HI4_CHECK_OK',json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pretrained',required=True);p.add_argument('--data-root',default='')
    main(p.parse_args())
