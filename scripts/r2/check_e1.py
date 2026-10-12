"""Separate, six-image E1 preflight; no formal training output is written."""
import argparse
import gc
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config-file',default=str(REPO/'configs/r2_m5_flat_tail.yml'))
    ap.add_argument('--output',required=True)
    ap.add_argument('--resources',help='Optional explicit local path mapping JSON')
    ap.add_argument('--local-diagnostic-support',help='Local-only torch1.13 diagnostic helper directory; never used by server launcher')
    args=ap.parse_args()
    helper=None
    if args.local_diagnostic_support:
        sys.path.insert(0,args.local_diagnostic_support)
        import common as helper
        sys.path.insert(0,str(REPO))
    from config import cfg
    from train import set_seed
    from model import make_model
    from loss import make_loss
    from loss.r2_kd import flat_tail_kd
    from model.m2_teacher import normalized_clip_logits, kd_kl
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from utils.m5_checkpoint import attach_metadata, load_teacher, save_training, restore_training, compare_config
    from utils.r2_experiment import config_differences, tensor_digest, actual_parent, write_json, verify_parent_artifacts
    from utils.runtime import runtime_summary
    from tools.m2.artifacts import validate_text_bank, file_digest
    c=cfg.clone()
    c.merge_from_file(args.config_file)
    local_paths=None
    if args.resources:
        local_paths=json.loads(Path(args.resources).read_text(encoding='utf-8'))
        c.DATASETS.ROOT_DIR=str(Path(local_paths['dataset']).parent)
        c.DATASETS.SUBDIR=Path(local_paths['dataset']).name
        c.MODEL.PRETRAIN_PATH=c.M2.CLIP_PATH=local_paths['clip']
        c.M2.PROMPT_WEIGHT=local_paths['m5_prompt']
        c.M2.TEXT_BANK=local_paths['m5_text']
        c.M2.TEACHER_WEIGHT=local_paths['m5_teacher']
    import timm, torchvision
    if helper is None:
        assert sys.version_info[:2]==(3,8),sys.version
        assert torch.__version__.split('+')[0]=='2.4.1' and torch.version.cuda=='12.1'
        assert torchvision.__version__.split('+')[0]=='0.13.1' and timm.__version__=='1.0.19'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    print(runtime_summary(),flush=True)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    result=dict(status='RUNNING',local_only=helper is not None,formal_training_started=False,
                preflight_images=6,teacher_updates=0,prompt_updates=0)
    try:
        result['config_diff_vs_parent']=config_differences(c,allow_path_mapping=True)
        result['parent_artifacts_verified']=verify_parent_artifacts(c)
        # Enabling E1 adds no model parameters, RNG draws or initialization changes.
        disabled=c.clone()
        disabled.R2.ENABLED=False
        set_seed(c.SOLVER.SEED,c.SOLVER.CUDNN_BENCHMARK)
        baseline=make_model(disabled,500,7,3)
        initial_hash=tensor_digest(baseline.state_dict())
        baseline_rng=torch.get_rng_state().clone()
        baseline_keys=list(baseline.state_dict())
        del baseline
        gc.collect()
        set_seed(c.SOLVER.SEED,c.SOLVER.CUDNN_BENCHMARK)
        model=make_model(c,500,7,3)
        assert tensor_digest(model.state_dict())==initial_hash
        assert torch.equal(torch.get_rng_state(),baseline_rng)
        assert list(model.state_dict())==baseline_keys
        result['initial_state_matches_disabled_M5']=True
        result['initial_state_sha256']=initial_hash
        result['construction_rng_matches']=True
        old=actual_parent()
        with_defaults=old.clone()
        with_defaults.R2=disabled.R2.clone()
        compare_config(old.dump(),with_defaults.dump())
        bank=torch.load(c.M2.TEXT_BANK,map_location='cpu',weights_only=False)
        validate_text_bank(bank,c,500)
        if helper is None:
            teacher,teacher_obj=load_teacher(c,500,'cuda')
        else:
            assert local_paths is not None and torch.__version__.startswith('1.13.')
            # Compatibility is confined to this explicit local preflight.
            # Formal finetune and its load_teacher remain the original code.
            with torch.random.fork_rng(devices=[]):
                teacher,_=helper.teacher(local_paths,'cuda')
            teacher_obj=torch.load(c.M2.TEACHER_WEIGHT,map_location='cpu',weights_only=False)
        result['teacher_local_compatibility']=getattr(teacher,'diagnostic_compatibility',{})
        assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
        teacher_buffers={name:value.cpu().clone() for name,value in teacher.named_buffers()}
        text=bank['text_bank'].float().cuda().detach()
        model=model.cuda().train()
        loss_fn,center=make_loss(c,500)
        optimizer,optimizer_center=make_optimizer(c,model,center)
        scheduler=create_scheduler(c,optimizer,iters_per_epoch=1)
        scaler=torch.cuda.amp.GradScaler()
        attach_metadata(model,c,bank,teacher_obj)
        model.m5_metadata['r2_preflight_only']=True
        from datasets.whu_mars import WHU_MARS
        from PIL import Image
        import torchvision.transforms as T
        from timm.data.random_erasing import RandomErasing
        dataset=WHU_MARS(root=c.DATASETS.ROOT_DIR,modalities=c.DATASETS.MODALITIES,subdir=c.DATASETS.SUBDIR,verbose=False)
        ids=sorted(set.intersection(*[set(r[1] for r in dataset.train[m]) for m in c.DATASETS.MODALITIES]))[:2]
        target=torch.tensor(ids,device='cuda',dtype=torch.long)
        transform=T.Compose([T.Resize(c.INPUT.SIZE_TRAIN,interpolation=3),T.RandomHorizontalFlip(c.INPUT.PROB),
            T.Pad(c.INPUT.PADDING),T.RandomCrop(c.INPUT.SIZE_TRAIN),T.ToTensor(),
            T.Normalize(c.INPUT.PIXEL_MEAN,c.INPUT.PIXEL_STD),
            RandomErasing(probability=c.INPUT.RE_PROB,mode='pixel',max_count=1,device='cpu')])
        imgs,cams,selected=[],[],[]
        for m in c.DATASETS.MODALITIES:
            rows=[next(r for r in dataset.train[m] if r[1]==y) for y in ids]
            images=[]
            for row in rows:
                with Image.open(row[0]) as im:
                    images.append(transform(im.convert('RGB')))
                selected.append(dict(path=row[0],train_pid=row[1],camid=row[2],modality=m))
            imgs.append(torch.stack(images).cuda())
            cams.append(torch.tensor([r[2] for r in rows],device='cuda'))
        target_rep=target.repeat(3)
        x_all=torch.cat(imgs)
        modality=torch.arange(3,device='cuda').repeat_interleave(len(target))
        cam_all=torch.cat(cams)
        platform=((cam_all==5)|(cam_all==6)).long()
        scheduler.step(1)
        skips=0
        gradient_checks=[]
        for attempt in range(8):
            scheduler.step_update(attempt)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                score,pre,_=model(imgs,target,cams)
                base_loss,ce,triplet=loss_fn(score,pre,target_rep)
                with torch.no_grad():
                    teacher_pre=teacher.forward_pre_bn(x_all,modality,platform)
                    lt=normalized_clip_logits(teacher_pre,teacher.base.clip_proj,text,c.M2.TEXT_TEMPERATURE)
                ls=normalized_clip_logits(pre,model.base.clip_proj,text,c.M2.TEXT_TEMPERATURE)
                kd,stats=flat_tail_kd(lt,ls,target_rep,c.M2.KD_TEMPERATURE,True)
                total=base_loss+c.M2.KD_WEIGHT*kd
            assert torch.isfinite(total) and kd.dtype==torch.float32
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradients={}
            finite=True
            for name in ('base.patch_embed.proj.weight','base.token_trajectory.gain'):
                grad=dict(model.named_parameters())[name].grad
                assert grad is not None,name
                isfinite=bool(torch.isfinite(grad).all())
                gradient_checks.append(dict(attempt=attempt+1,parameter=name,finite=isfinite,
                    scale=scaler.get_scale()))
                finite=finite and isfinite
                if isfinite:
                    assert grad.abs().sum()>0,name
                    gradients[name]=grad.norm().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            before=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()<before:
                skips+=1
                print('PREFLIGHT_AMP_SKIP',attempt+1,'scale',before,'->',scaler.get_scale(),flush=True)
                continue
            assert finite,'Nonfinite gradients were not skipped by GradScaler'
            break
        else:
            raise AssertionError('No successful AMP update within 8 preflight attempts')
        assert model.base.clip_proj.grad is None and not model.base.clip_proj.requires_grad
        assert all(p.grad is None for p in teacher.parameters()) and not text.requires_grad
        assert all(torch.equal(value.cpu(),teacher_buffers[name]) for name,value in teacher.named_buffers())
        result.update(gradients=gradients,teacher_frozen_eval_buffers_unchanged=True,
            actual_iterations=attempt+1,actual_updates=1,AMP_skips=skips,gradient_checks=gradient_checks,
            CE=ce.item(),Triplet=triplet.item(),KD=kd.item(),stats=stats,
            target_rep=target_rep.tolist(),modality=modality.tolist(),platform=platform.tolist(),images=selected)
        model.eval()
        with torch.no_grad():
            expected=model(x_all,mode=1).cpu()
        saved=save_training(str(out),'transformer',0,model,optimizer,scheduler,scaler,attempt+1,1,skips,0.)
        full=torch.load(out/'transformer_0.pth',map_location='cpu',weights_only=False)
        deploy=torch.load(out/'transformer_0_student.pth',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,deploy['state_dict'][k]) for k,v in full['state_dict'].items())
        expected_random=(torch.rand(3),np.random.rand(3),random.random())
        with torch.no_grad():
            model.classifier.weight.add_(1.)
        try:
            restore_training(str(out/'transformer_0.pth'),model,optimizer,scheduler,scaler)
        except ValueError as exc:
            assert 'preflight checkpoint' in str(exc)
        else:
            raise AssertionError('Preflight checkpoint accepted as formal resume')
        restored=restore_training(str(out/'transformer_0.pth'),model,optimizer,scheduler,scaler,allow_preflight=True)
        assert restored[:4]==(1,attempt+1,1,skips)
        assert torch.equal(torch.rand(3),expected_random[0])
        assert np.array_equal(np.random.rand(3),expected_random[1]) and random.random()==expected_random[2]
        with torch.no_grad():
            torch.testing.assert_close(model(x_all,mode=1).cpu(),expected,rtol=0,atol=0)
        model.load_param(str(out/'transformer_0_student.pth'))
        with torch.no_grad():
            torch.testing.assert_close(model(x_all,mode=1).cpu(),expected,rtol=0,atol=0)
        wrong=out/'wrong_mode.pth'
        torch.save({'metadata':{'r2':dict(experiment='m5_one_sided',kd_mode='one_sided_target',parent_code_commit=c.R2.PARENT_CODE_COMMIT)}},wrong)
        try:
            restore_training(str(wrong),model,optimizer,scheduler,scaler)
        except ValueError as exc:
            assert 'R2 resume' in str(exc)
        else:
            raise AssertionError('Wrong-mode resume accepted')
        if local_paths:
            try:
                restore_training(local_paths['m5_student'],model,optimizer,scheduler,scaler)
            except ValueError as exc:
                assert 'R2 resume' in str(exc)
            else:
                raise AssertionError('Historical M5 epoch60 accepted as E1 resume')
        result.update(status='LOCAL_PREFLIGHT_COMPLETED_WITH_CAVEAT' if helper else 'PREFLIGHT_COMPLETED',
            strict_state_reload=True,deployment_equal_full=True,features_equal_after_reload=True,
            resume_rng_optimizer_scheduler_scaler_checked=True,wrong_mode_resume_rejected=True,
            preflight_cannot_resume_formal_training=True,
            gpu_peak_bytes=torch.cuda.max_memory_allocated(),runtime=runtime_summary(),
            teacher_sha256=file_digest(c.M2.TEACHER_WEIGHT),text_sha256=file_digest(c.M2.TEXT_BANK))
        print('E1_PREFLIGHT',result['status'],flush=True)
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc))
        raise
    finally:
        write_json(out/'preflight.json',result)


if __name__=='__main__':
    main()
