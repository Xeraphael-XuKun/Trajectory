"""Independent E3 preflight: exact five groups, real AMP and full restore."""
import argparse,gc,inspect,json,random,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


def check_restored_states(full,optimizer,scheduler,scaler):
    current=optimizer.state_dict()
    assert current['param_groups']==full['optimizer']['param_groups']
    assert current['state'].keys()==full['optimizer']['state'].keys()
    for index,saved in full['optimizer']['state'].items():
        for key,value in saved.items():
            actual=current['state'][index][key]
            if torch.is_tensor(value):
                assert torch.equal(actual.cpu(),value.cpu())
            else:
                assert actual==value
    assert scheduler.state_dict()==full['scheduler']
    assert scaler.state_dict()==full['scaler']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config-file',default=str(REPO/'configs/r2_m3_no_wd.yml'))
    ap.add_argument('--output',required=True)
    ap.add_argument('--resources')
    ap.add_argument('--local-loader-compat',action='store_true',help='Explicit local torch1.13 keyword adapter; never passed by server')
    args=ap.parse_args()
    loader_adapter_used=False
    if args.local_loader_compat:
        assert torch.__version__.startswith('1.13.')
    if args.local_loader_compat and 'weights_only' not in inspect.signature(torch.load).parameters:
        loader_adapter_used=True
        original_load=torch.load
        def load(*a,**kw):
            kw.pop('weights_only',None)
            return original_load(*a,**kw)
        torch.load=load
    import timm,torchvision
    if not args.local_loader_compat:
        assert sys.version_info[:2]==(3,8)
        assert torch.__version__.split('+')[0]=='2.4.1' and torch.version.cuda=='12.1'
        assert torchvision.__version__.split('+')[0]=='0.13.1' and timm.__version__=='1.0.19'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    from config import cfg
    from train import set_seed
    from model import make_model
    from loss import make_loss
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from processor.processor import deep_text_identity_loss
    from utils.m3_checkpoint import save_training,restore_training
    from utils.r2_experiment import config_differences,verify_parent_artifacts,tensor_digest,write_json,gradient_snapshot
    from utils.runtime import runtime_summary
    c=cfg.clone();c.merge_from_file(args.config_file)
    local=None
    if args.resources:
        local=json.loads(Path(args.resources).read_text(encoding='utf-8'))
        c.DATASETS.ROOT_DIR=str(Path(local['dataset']).parent);c.DATASETS.SUBDIR=Path(local['dataset']).name
        c.MODEL.PRETRAIN_PATH=c.M2.CLIP_PATH=local['clip']
        c.MODEL.TEXT_CLIP_PATH=local['clip']
        c.M2.TEXT_BANK=c.M2.ID_PROMPT_WEIGHT=c.M2.DEEP_CONTEXT_INIT=local['m3_bank']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=dict(status='RUNNING',formal_training_started=False,new_prompt_updates=0,
                conditional_teacher='not_applicable',local_loader_compat=args.local_loader_compat,
                load_keyword_adapter_used=loader_adapter_used)
    try:
        result['config_diff']=config_differences(c,True)
        result['parent_artifacts']=verify_parent_artifacts(c)
        off=c.clone();off.R2.ENABLED=False
        set_seed(c.SOLVER.SEED,c.SOLVER.CUDNN_BENCHMARK)
        baseline=make_model(off,500,7,3)
        initial=tensor_digest(baseline.state_dict());rng=torch.get_rng_state().clone()
        trainable=[n for n,p in baseline.named_parameters() if p.requires_grad]
        del baseline;gc.collect()
        set_seed(c.SOLVER.SEED,c.SOLVER.CUDNN_BENCHMARK)
        model=make_model(c,500,7,3)
        assert tensor_digest(model.state_dict())==initial and torch.equal(rng,torch.get_rng_state())
        assert trainable==[n for n,p in model.named_parameters() if p.requires_grad]
        result.update(initial_state_sha256=initial,initial_state_rng_trainability_matches_parent=True)
        center=torch.nn.Linear(1,1)
        old_opt,_=make_optimizer(off,model,center)
        optimizer,_=make_optimizer(c,model,center)
        assert old_opt.defaults==optimizer.defaults and len(old_opt.param_groups)==len(optimizer.param_groups)
        names={id(p):n for n,p in model.named_parameters()};changes=[]
        for old,new in zip(old_opt.param_groups,optimizer.param_groups):
            assert all(a is b for a,b in zip(old['params'],new['params']))
            for key in old:
                if key not in ('params','weight_decay'):assert old[key]==new[key],key
            if old['weight_decay']!=new['weight_decay']:
                changes.append(names[id(new['params'][0])])
                assert old['weight_decay']==1e-4 and new['weight_decay']==0 and new['lr']==3.5e-4
        assert set(changes)=={'base.deep_c0_controller.'+name for name in ('context','q.weight','k.weight','v.weight','o.weight')}
        assert sum(dict(model.named_parameters())[n].numel() for n in changes)==186368
        assert not optimizer.state and not old_opt.state
        result['only_optimizer_changes']=changes
        del old_opt,optimizer,center;gc.collect()
        from datasets.whu_mars import WHU_MARS
        from PIL import Image
        import torchvision.transforms as T
        from timm.data.random_erasing import RandomErasing
        dataset=WHU_MARS(root=c.DATASETS.ROOT_DIR,modalities=c.DATASETS.MODALITIES,subdir=c.DATASETS.SUBDIR,verbose=False)
        ids=sorted(set.intersection(*[set(r[1] for r in dataset.train[m]) for m in c.DATASETS.MODALITIES]))[:2]
        transform=T.Compose([T.Resize(c.INPUT.SIZE_TRAIN,interpolation=3),T.RandomHorizontalFlip(c.INPUT.PROB),
            T.Pad(c.INPUT.PADDING),T.RandomCrop(c.INPUT.SIZE_TRAIN),T.ToTensor(),
            T.Normalize(c.INPUT.PIXEL_MEAN,c.INPUT.PIXEL_STD),
            RandomErasing(probability=c.INPUT.RE_PROB,mode='pixel',max_count=1,device='cpu')])
        imgs,cams,selected=[],[],[]
        for m in c.DATASETS.MODALITIES:
            rows=[next(r for r in dataset.train[m] if r[1]==y and ((r[2] in (5,6))==(i==1))) for i,y in enumerate(ids)]
            images=[]
            for row in rows:
                with Image.open(row[0]) as im:images.append(transform(im.convert('RGB')))
                selected.append(dict(path=row[0],pid=row[1],camid=row[2],modality=m))
            imgs.append(torch.stack(images).cuda());cams.append(torch.tensor([r[2] for r in rows],device='cuda'))
        target=torch.tensor(ids,device='cuda');target_rep=target.repeat(3);x_all=torch.cat(imgs)
        model=model.cuda().eval();controller=model.base.deep_c0_controller
        with torch.no_grad():
            controller.r2_collect_stats=False
            before=model(x_all,mode=1)
            controller.r2_collect_stats=True;controller.r2_last_stats=[]
            after=model(x_all,mode=1)
            assert torch.equal(before,after) and len(controller.r2_last_stats)==11
        model.train()
        assert not model.deep_identity.text.training and not model.deep_identity.id_bank.requires_grad
        frozen=[p for p in model.deep_identity.text.parameters()]
        assert all(not p.requires_grad for p in frozen) and not model.base.clip_proj.requires_grad
        loss_fn,center=make_loss(c,500)
        optimizer,_=make_optimizer(c,model,center)
        scheduler=create_scheduler(c,optimizer,iters_per_epoch=1);scheduler.step(1)
        scaler=torch.cuda.amp.GradScaler()
        model.m2_metadata['r2_preflight_only']=True
        skips=0
        for attempt in range(8):
            optimizer.zero_grad();controller.r2_last_stats=[]
            with torch.cuda.amp.autocast(enabled=True):
                score,pre,_,aux=model(imgs,target,cams)
                base,ce,triplet=loss_fn(score,pre,target_rep)
                extra,stats=deep_text_identity_loss(aux,c.M2.TEMPERATURE,c.M2.TEXT_ID_WEIGHT,c.M2.TEXT_ANCHOR_WEIGHT)
                total=base+extra
            assert torch.equal(aux['targets'],target_rep) and aux['text'].dtype==torch.float32
            assert torch.isfinite(total)
            scaler.scale(total).backward();scaler.unscale_(optimizer)
            gradients=gradient_snapshot(model)
            for name in ('base.patch_embed.proj.weight','base.token_trajectory.gain','base.deep_c0_controller.context'):
                grad=dict(model.named_parameters())[name].grad
                assert grad is not None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<scale:
                skips+=1;print('PREFLIGHT_AMP_SKIP',attempt+1,scale,'->',scaler.get_scale(),flush=True);continue
            for name in ('base.patch_embed.proj.weight','base.token_trajectory.gain','base.deep_c0_controller.context'):
                grad=dict(model.named_parameters())[name].grad
                assert torch.isfinite(grad).all() and grad.abs().sum()>0,name
            break
        else:raise AssertionError('No successful AMP update within 8 attempts')
        assert all(p.grad is None for p in frozen) and model.deep_identity.id_bank.grad is None and model.base.clip_proj.grad is None
        model.eval();controller.r2_collect_stats=False
        with torch.no_grad():
            expected=model(x_all,mode=1).cpu()
            _,_,_,fixed_aux=model(imgs,target,cams)
            expected_loss,_=deep_text_identity_loss(fixed_aux,c.M2.TEMPERATURE,c.M2.TEXT_ID_WEIGHT,c.M2.TEXT_ANCHOR_WEIGHT)
            expected_loss=expected_loss.cpu()
        save_training(str(out),'transformer',0,model,optimizer,scheduler,scaler,attempt+1,1,skips)
        full=torch.load(out/'transformer_0.pth',map_location='cpu',weights_only=False)
        deployment=torch.load(out/'transformer_0_student.pth',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,full['state_dict'][k]) for k,v in deployment['state_dict'].items())
        random_after=(torch.rand(3),np.random.rand(3),random.random())
        with torch.no_grad():model.classifier.weight.add_(1.)
        try:restore_training(str(out/'transformer_0.pth'),model,optimizer,scheduler,scaler)
        except ValueError as e:assert 'preflight checkpoint' in str(e)
        else:raise AssertionError('preflight accepted for formal resume')
        restored=restore_training(str(out/'transformer_0.pth'),model,optimizer,scheduler,scaler,allow_preflight=True)
        assert restored==(1,attempt+1,1,skips)
        check_restored_states(full,optimizer,scheduler,scaler)
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),full['rng']['cuda']))
        assert torch.equal(torch.rand(3),random_after[0]) and np.array_equal(np.random.rand(3),random_after[1]) and random.random()==random_after[2]
        with torch.no_grad():
            assert torch.equal(model(x_all,mode=1).cpu(),expected)
            _,_,_,fixed_aux=model(imgs,target,cams)
            restored_loss,_=deep_text_identity_loss(fixed_aux,c.M2.TEMPERATURE,c.M2.TEXT_ID_WEIGHT,c.M2.TEXT_ANCHOR_WEIGHT)
            assert torch.equal(restored_loss.cpu(),expected_loss)
        wrong=out/'wrong_mode.pth';torch.save({'metadata':{'r2':dict(experiment='other')}},wrong)
        try:restore_training(str(wrong),model,optimizer,scheduler,scaler)
        except ValueError as e:assert 'R2 resume' in str(e)
        else:raise AssertionError('wrong experiment accepted')
        if local:
            try:restore_training(local['m3_student'],model,optimizer,scheduler,scaler)
            except ValueError as e:assert 'R2 resume' in str(e)
            else:raise AssertionError('old M3 accepted as E3 resume')
        # Strict deployment reload retains all five controller tensors/context,
        # and needs no text tower or bank at inference.
        del full,deployment,fixed_aux,aux;model=model.cpu();gc.collect();torch.cuda.empty_cache()
        deploy_cfg=c.clone();deploy_cfg.M2.DEPLOY_ONLY=True;deploy_cfg.MODEL.PRETRAIN_CHOICE='no'
        deploy_cfg.M2.TEXT_BANK=deploy_cfg.M2.CLIP_PATH='/not/read/in/deployment'
        deploy_model=make_model(deploy_cfg,500,7,3).eval()
        payload=torch.load(out/'transformer_0_student.pth',map_location='cpu',weights_only=False)
        deploy_model.load_state_dict(payload['state_dict'],strict=True)
        deploy_model.load_param(str(out/'transformer_0_student.pth'))
        deploy_model=deploy_model.cuda()
        with torch.no_grad():assert torch.equal(deploy_model(x_all,mode=1).cpu(),expected)
        result.update(status='LOCAL_PREFLIGHT_COMPLETED_WITH_CAVEAT' if args.local_loader_compat else 'PREFLIGHT_COMPLETED',
            actual_iterations=attempt+1,actual_updates=1,AMP_skips=skips,controller_gradients=gradients,
            CE=ce.item(),Triplet=triplet.item(),text_identity=stats['loss_it'].item(),anchor=stats['loss_anchor'].item(),
            images=selected,observations_preserve_real_forward=True,frozen_text_and_ids=True,
            strict_full_reload=True,strict_deployment_reload=True,deployment_equal_full=True,
            features_equal_after_reload=True,text_loss_equal_after_reload=True,
            resume_rng_optimizer_scheduler_scaler=True,old_m3_wrong_mode_preflight_resume_rejected=True,
            gpu_peak_bytes=torch.cuda.max_memory_allocated(),runtime=runtime_summary())
        print('E3_PREFLIGHT',result['status'],flush=True)
    except Exception as e:
        result.update(status='FAILED',error=repr(e));raise
    finally:write_json(out/'preflight.json',result)


if __name__=='__main__':main()
