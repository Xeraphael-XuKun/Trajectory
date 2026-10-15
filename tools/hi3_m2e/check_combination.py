"""HI3 plus original M2-e: real artifacts, gradient and deployment smoke check."""
import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from tools.m2.artifacts import load_config, validate_text_bank, seed_stage
from model import make_model
from model.m2_teacher import normalized_clip_logits, kd_kl
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.m5_checkpoint import load_teacher, attach_metadata, save_training, restore_training


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config_file',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-per-modality',type=int,default=2)
    p.add_argument('--output',required=True)
    p.add_argument('--reference-hi3',default='')
    p.add_argument('opts',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    if a.batch_per_modality<2 or a.batch_per_modality%2:
        raise ValueError('Use an even batch per modality, at least two identities')
    torch.set_num_threads(4)
    c=load_config(a.config_file,a.opts)
    seed_stage(c)
    bank=torch.load(c.M2.TEXT_BANK,map_location='cpu',weights_only=False)
    validate_text_bank(bank,c)
    classes=len(bank['text_bank'])
    model=make_model(c,classes,0,0)
    assert model.base.token_trajectory is None
    assert model.base.history_adapter is not None
    assert not model.base.clip_proj.requires_grad
    # Check initialization and nonzero-residual execution against original HI3.
    reference_check='not requested'
    if a.reference_hi3:
        ref=torch.load(a.reference_hi3,map_location='cpu',weights_only=False)
        own=model.state_dict()
        assert own.keys()==ref['state_dict'].keys(), (set(own)-set(ref['state_dict']),set(ref['state_dict'])-set(own))
        for key,value in own.items():
            assert torch.equal(value,ref['state_dict'][key]),key
        model.eval()
        with torch.no_grad():
            x=ref['input']
            torch.testing.assert_close(model(x,mode=1),ref['post_zero'],rtol=0,atol=0)
            model.base.history_adapter.gain.copy_(ref['nonzero_gain'])
            torch.testing.assert_close(model(x,mode=1),ref['post_nonzero'],rtol=0,atol=0)
            model.base.history_adapter.gain.zero_()
        reference_check='all state tensors, zero/nonzero-gain post output exactly equal'
        del ref,own
        gc.collect()
    device=torch.device(a.device)
    model.to(device)
    text=bank['text_bank'].float().to(device).detach()
    teacher,obj=load_teacher(c,classes,device)
    assert teacher.base.history_adapter is None and teacher.base.token_trajectory is None
    assert not teacher.training and all(not v.requires_grad for v in teacher.parameters())
    attach_metadata(model,c,bank,obj)
    del obj
    gc.collect()
    center=torch.nn.Linear(1,1).to(device)
    optimizer,_=make_optimizer(c,model,center)
    by_id={id(group['params'][0]):group for group in optimizer.param_groups}
    for name,param in model.base.history_adapter.named_parameters():
        group=by_id[id(param)]
        assert group['lr']==.00035 and group['weight_decay']==.0001,(name,group['lr'],group['weight_decay'])
    assert id(model.base.clip_proj) not in by_id
    n=a.batch_per_modality
    # Deterministic synthetic input, with each identity present in all modalities.
    joined=torch.linspace(-1.,1.,3*n*3*256*128,device=device).reshape(3*n,3,256,128)
    images=list(joined.split(n))
    cameras=[torch.tensor(([0,5]*(n//2)),device=device) for _ in range(3)]
    target=torch.arange(n,device=device).remainder(2).repeat(3)
    mods=torch.arange(3,device=device).repeat_interleave(n)
    cams=torch.cat(cameras)
    platforms=((cams==5)|(cams==6)).long()
    model.train()
    count_before=int(model.bottleneck.num_batches_tracked)
    score,g,_,aux=model(images,camids=cameras,history_diagnostics=True)
    assert int(model.bottleneck.num_batches_tracked)==count_before+1
    with torch.no_grad():
        gt=teacher.forward_pre_bn(joined,mods,platforms)
        lt=normalized_clip_logits(gt,teacher.base.clip_proj,text,c.M2.TEXT_TEMPERATURE)
    ls=normalized_clip_logits(g,model.base.clip_proj,text,c.M2.TEXT_TEMPERATURE)
    kd=kd_kl(lt,ls,c.M2.KD_TEMPERATURE)
    pred=history_prediction_loss(aux)
    subset=[model.base.history_adapter.gain,model.bottleneck.weight,model.classifier.weight]
    kd_grad=torch.autograd.grad(kd,subset,retain_graph=True,allow_unused=True)
    assert kd_grad[0] is not None and torch.isfinite(kd_grad[0]).all() and kd_grad[0].norm()>0
    assert kd_grad[1] is None and kd_grad[2] is None
    pred_grad=torch.autograd.grad(pred,[subset[0],model.base.history_adapter.predictors[0][0].weight],retain_graph=True,allow_unused=True)
    assert pred_grad[0] is None and pred_grad[1].norm()>0
    assert ls.shape==(3*n,classes) and lt.shape==ls.shape
    torch.testing.assert_close(kd, (c.M2.KD_TEMPERATURE**2)*F.kl_div(
        (ls/c.M2.KD_TEMPERATURE).log_softmax(-1),
        (lt/c.M2.KD_TEMPERATURE).softmax(-1),reduction='batchmean'),rtol=1e-4,atol=1e-6)
    del score,g,aux,ls,lt,kd,pred,kd_grad,pred_grad
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    scheduler=create_scheduler(c,optimizer,iters_per_epoch=1)
    updates=skips=0
    for attempt in range(16):
        optimizer.zero_grad()
        scheduler.step_update(attempt)
        with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
            score,g,_,aux=model(images,camids=cameras,history_diagnostics=True)
            with torch.no_grad():
                gt=teacher.forward_pre_bn(joined,mods,platforms)
                lt=normalized_clip_logits(gt,teacher.base.clip_proj,text,c.M2.TEXT_TEMPERATURE)
            ls=normalized_clip_logits(g,model.base.clip_proj,text,c.M2.TEXT_TEMPERATURE)
            kd=kd_kl(lt,ls,c.M2.KD_TEMPERATURE)
            pred=history_prediction_loss(aux)
            ce=F.cross_entropy(score,target)
            tri=TripletLoss()(g,target)[0]
            loss=ce+tri+c.HISTORY.LOSS_WEIGHT*pred+c.M2.KD_WEIGHT*kd
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scale=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale()<scale:
            skips+=1
        else:
            updates+=1
            break
    assert updates==1 and model.base.history_adapter.gain.norm()>0
    assert all(p.grad is None for p in teacher.parameters())
    assert model.base.clip_proj.grad is None
    # An actual post-step state can reload for deployment with no teacher/text.
    model.eval()
    with tempfile.TemporaryDirectory(prefix='hi3-m2e-check-') as temp:
        save_training(temp,'transformer',1,model,optimizer,scheduler,scaler,attempt+1,updates,skips,0.)
        restored=restore_training(str(Path(temp)/'transformer_1.pth'),model,optimizer,scheduler,scaler)
        assert restored[:4]==(2,attempt+1,updates,skips)
        deploy_cfg=c.clone()
        deploy_cfg.defrost()
        deploy_cfg.M2.ENABLED=False
        deploy_cfg.MODEL.PRETRAIN_CHOICE='no'
        deploy_cfg.M2.TEXT_BANK='unavailable-at-deployment.pt'
        deploy_cfg.M2.TEACHER_WEIGHT='unavailable-at-deployment.pth'
        deployment=make_model(deploy_cfg,classes,0,0).to(device).eval()
        deployment.load_param(str(Path(temp)/'transformer_1_student.pth'))
        with torch.no_grad():
            for neck in ['after','before']:
                deployment.neck_feat=model.neck_feat=neck
                torch.testing.assert_close(model(joined[:2],mode=1),deployment(joined[:2],mode=1),rtol=0,atol=0)
        assert not any('lora' in k or 'text_bank' in k for k in deployment.state_dict())
        deployment.base.history_adapter.specification[5]=1.
        try:
            deployment.load_param(str(Path(temp)/'transformer_1_student.pth'))
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong History alpha was accepted')
    report=dict(status='PASSED_COMBINATION_SMOKE',device=str(device),
        torch=torch.__version__,batch=3*n,classes=classes,synthetic_input=True,
        training_attempts=attempt+1,optimizer_updates=updates,amp_skips=skips,
        losses=dict(ce=float(ce),triplet=float(tri),prediction=float(pred),kd=float(kd)),
        original_hi3_equivalence=reference_check,
        actual_original_teacher=True,teacher_has_history=False,
        history_lr=.00035,history_weight_decay=.0001,
        kd_gradients='gain nonzero; BN/classifier None; teacher/projection frozen',
        prediction_gradients='predictor nonzero; gain None',
        reload='full optimizer/scheduler/scaler/RNG and teacher-free pre/post exact',
        limitations='Not formal 60-epoch training or retrieval. Synthetic batch; no performance claim.')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/'implementation_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
