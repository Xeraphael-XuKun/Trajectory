"""Focused four-cell mechanism checks, without formal training."""
import json
import subprocess
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from config import cfg
from model.backbones.history_adapter import HistoryInnovationAdapter
from loss.history_prediction import history_prediction_loss


def build(scope='patch', mode='tanh', cls=HistoryInnovationAdapter):
    torch.manual_seed(1235)
    if cls is HistoryInnovationAdapter:
        return cls(16,3,4,4,0.25,[2,3],scope,mode)
    return cls(16,3,4,4,0.25,[2,3])


def main():
    torch.set_num_threads(4)
    source=subprocess.check_output(['git','-C',str(ROOT),'show','b0273d3:model/backbones/history_adapter.py']).decode()
    old=types.ModuleType('old_adapter'); exec(compile(source,'old_adapter','exec'),old.__dict__)
    ref=build(cls=old.HistoryInnovationAdapter); current=build()
    assert ref.state_dict().keys()==current.state_dict().keys()
    assert all(torch.equal(v,current.state_dict()[k]) for k,v in ref.state_dict().items())
    with torch.no_grad():
        ref.gain.fill_(0.4); current.gain.fill_(0.4)
    h=torch.randn(2,5,16)
    hist0,hist1,prev0,prev1=[],[],None,None
    for l in range(3):
        r0,hist0,prev0,item0=ref.correction(h,l,hist0,prev0,True,True)
        r1,hist1,prev1,item1=current.correction(h,l,hist1,prev1,True,True)
        assert torch.equal(r0,r1) and torch.equal(prev0,prev1)
        if l: assert torch.equal(item0['prediction'],item1['prediction'])
        h=h+r0+0.01

    bounded,linear=build(),build(mode='linear')
    h=torch.randn(2,5,16)
    rb,_,_,_=bounded.correction(h,0,[],None)
    rl,_,_,_=linear.correction(h,0,[],None)
    assert torch.equal(rb,rl)
    rb.sum().backward(); rl.sum().backward()
    assert torch.equal(bounded.gain.grad,linear.gain.grad)
    assert bounded.gain.grad.abs().sum()>0
    with torch.no_grad(): bounded.gain.fill_(0.7); linear.gain.fill_(0.7)
    rb,_,_,_=bounded.correction(h,0,[],None)
    rl,_,_,_=linear.correction(h,0,[],None)
    assert torch.allclose(rl,rb*(0.7/torch.tanh(torch.tensor(0.7))),rtol=1e-5,atol=1e-7)

    patches=build(); both=build(scope='all')
    assert all(torch.equal(v,both.state_dict()[k]) for k,v in patches.state_dict().items() if k!='specification')
    with torch.no_grad(): patches.gain.fill_(0.2); both.gain.fill_(0.2); both.cls_gain.fill_(0.3)
    h=torch.randn(2,5,16,requires_grad=True)
    rp,_,_,_=patches.correction(h,0,[],None)
    ra,hist,prev,_=both.correction(h,0,[],None,True)
    assert torch.equal(rp[:,1:],ra[:,1:]) and ra[:,:1].abs().sum()>0
    assert len(hist)==1 and hist[0].shape[1]==5 and prev.shape[1]==5
    assert not hist[0].requires_grad and not prev.requires_grad
    r,_,_,item=both.correction(h*1.1,1,hist,prev,True)
    changed_h=h.detach().clone(); changed_h[:,:1]=torch.randn_like(changed_h[:,:1])*3
    changed_hist=[v.clone() for v in hist]; changed_hist[0][:,:1]+=0.7
    changed_prev=prev.clone();changed_prev[:,:1]-=0.5
    changed_h[:,1:]=h.detach()[:,1:]*1.1
    r2,_,_,item2=both.correction(changed_h,1,changed_hist,changed_prev,True)
    assert torch.equal(r[:,1:],r2[:,1:]) and not torch.equal(r[:,:1],r2[:,:1])
    assert item['prediction'].shape[1]==4 and item['target'].shape[1]==4
    loss=history_prediction_loss({'layers':{2:item},'prediction_layers':[2]})
    loss2=history_prediction_loss({'layers':{2:item2},'prediction_layers':[2]})
    assert torch.equal(loss,loss2)
    loss.backward()
    assert both.cls_gain.grad is None and both.gain.grad is None and h.grad is None
    assert not torch.equal(bounded.specification,linear.specification)
    assert not torch.equal(patches.specification,both.specification)

    configs={}
    for group in ['HI0','HI1','HI2','HI3']:
        c=cfg.clone();c.merge_from_file(str(ROOT/f'configs/history_{group}.yml'));configs[group]=c
    for group,field,value in [('HI0','LOSS_WEIGHT',0.0),('HI2','TOKEN_SCOPE','all'),('HI3','GAIN_MODE','linear')]:
        actual=configs[group].clone();expected=configs['HI1'].clone()
        expected.HISTORY[field]=value;expected.OUTPUT_DIR=actual.OUTPUT_DIR
        assert expected.dump()==actual.dump()
    report={'old_hi1_nonzero_forward_exact':True,'linear_vs_tanh_initial_gradient_equal':True,
            'linear_preserves_alpha':True,'cls_own_history_and_shared_network':True,
            'all_variants_prediction_loss_patch_only':True,'config_single_factor_differences':True,
            'inference_signature_distinguishes_variants':True,
            'bounded_gain_diagnostics':bounded.gain_diagnostics()}
    out=ROOT/'work/history_variants';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HISTORY_VARIANTS_CHECK_OK',json.dumps(report))


if __name__=='__main__': main()
