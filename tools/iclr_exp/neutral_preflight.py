"""Method sanity checks, actual old resources, small backward and full-state reload."""
import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F


def local_position_diagnostic_loader(c,classes,device):
    """Explicit preflight-only diagnostic. NEVER called by training/stat extraction."""
    from model.m2_teacher import ConditionalVisualTeacher
    from utils.m5_checkpoint import architecture
    from tools.m2.artifacts import VERSION, file_digest, validate_source
    obj=torch.load(c.M2.TEACHER_WEIGHT,map_location='cpu',weights_only=False);meta=obj['metadata']
    if (meta['implementation']!=VERSION or meta['variant']!=c.M2.VARIANT or meta['stage']!='teacher' or
        meta['epoch']!=60 or meta['teacher_architecture']!=architecture(c) or meta['text_sha256']!=file_digest(c.M2.TEXT_BANK)):
        raise ValueError('local diagnostic teacher metadata mismatch')
    validate_source(meta['source'],c)
    with torch.random.fork_rng(devices=[]):teacher=ConditionalVisualTeacher(c,classes)
    differences=[]
    for key,value in teacher.base.state_dict().items():
        if key.startswith('fc.'):continue
        saved=obj['state_dict']['base.'+key]
        if not torch.equal(value.cpu(),saved.cpu()):differences.append((key,float((value.cpu()-saved.cpu()).abs().max())))
    if (not str(torch.__version__).startswith('1.13.') or len(differences)!=1 or
        differences[0][0]!='pos_embed' or differences[0][1]>2e-8):
        raise ValueError('diagnostic allowance covers only verified PyTorch1.13 positional interpolation roundoff: '+str(differences))
    teacher.load_state_dict(obj['state_dict'],strict=True)
    teacher.to(device).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    print('LOCAL DIAGNOSTIC ONLY: frozen pos_embed difference',differences,flush=True)
    return teacher,obj


def main():
    p=argparse.ArgumentParser(description='I2 必要检查；本地可用明确标记的部分统计，服务器要求正式共享缓存')
    p.add_argument('--resources',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-per-modality',type=int,default=2,choices=[2,64])
    p.add_argument('--require-stats',action='store_true')
    p.add_argument('--allow-local-position-roundoff',action='store_true')
    a=p.parse_args();torch.set_num_threads(a.threads)
    from contextlib import ExitStack
    from unittest.mock import patch
    from tools.iclr_exp.configuration import load_config
    from tools.iclr_exp.neutral_assets import load_resources,load_shift,legacy_config,paired_mask
    from tools.iclr_exp.neutral_stats import estimate,CleanTrain
    from tools.iclr_exp.runtime import deployment_state,NEUTRAL_VERSION
    from tools.iclr_exp.neutral_runtime import I2Run
    from tools.iclr_exp.audit import isolated_eval,build_panel
    from tools.m2.artifacts import train_source,rng_state,restore_rng,file_digest
    from model.m2_teacher import normalized_clip_logits,kd_kl
    from loss.iclr_neutral import fixed_teacher_targets,neutral_kd
    from model import make_model
    from loss import make_loss
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from train import set_seed
    c=load_config(variant=ROOT/'configs/iclr_exp/m5_neutral.yml',resources=a.resources)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);checks={}
    def check(name,ok,evidence=True):
        if not ok:raise AssertionError(name+': '+str(evidence))
        checks[name]=evidence
    # Hand calculation uses unequal images/camera and unequal differences/identity.
    sums=torch.zeros(2,3,7,5,dtype=torch.float64);counts=torch.zeros(2,3,7,dtype=torch.long)
    vector=torch.tensor([1.,-1.,0.,0.,0.],dtype=torch.float64)
    for m in range(3):
        for y,cam,n,value in ((0,0,9,1),(0,1,1,3),(0,5,2,6),(1,0,1,2),(1,6,1,12)):
            sums[y,m,cam]=n*value*vector;counts[y,m,cam]=n
    computed=estimate(sums,counts,[100,200])
    check('camera_then_platform_then_ID_equal_weight',torch.equal(computed['shift_fp64'],(7*vector).repeat(3,1)))
    check('fit_mean_AG_gap_zero',max(computed['fit_residual_mean_norm'])<1e-12)
    check('crossfit_local_seed_repeat',computed['crossfit_fold_ids']==estimate(sums,counts,[100,200])['crossfit_fold_ids'])
    try:estimate(sums*0,counts*0,[100,200])
    except ValueError:check('no_paired_ID_blocks_estimation',True)
    else:check('no_paired_ID_blocks_estimation',False)
    gen=torch.Generator().manual_seed(20261012)
    logits=torch.randn(7,500,generator=gen);mods=torch.arange(7)%3;views=torch.tensor([0,0,0,0,0,1,1]);delta=torch.randn(3,500,generator=gen);delta-=delta.mean(-1,keepdim=True)
    lp,lq=fixed_teacher_targets(logits,mods,views,delta)
    expected=(logits+(.5-views.float())[:,None]*delta[mods]).div(2).log_softmax(-1)
    check('neutral_sign_and_distill_temperature',torch.equal(lq,expected))
    lp2,lq2=fixed_teacher_targets(logits+3,mods,views,delta)
    check('common_scalar_softmax_invariance',torch.allclose(lp,lp2,atol=1e-6) and torch.allclose(lq,lq2,atol=1e-6))
    g=torch.randn(7,768,generator=gen,requires_grad=True);proj=torch.randn(768,512,generator=gen);text=F.normalize(torch.randn(500,512,generator=gen),dim=-1)
    raw,zero=fixed_teacher_targets(logits,mods,views,torch.zeros_like(delta))
    new=neutral_kd(g,proj,text,zero);old=kd_kl(logits,normalized_clip_logits(g,proj,text),2.)
    dn=torch.autograd.grad(new,g,retain_graph=True)[0];do=torch.autograd.grad(old,g)[0]
    check('zero_shift_old_M5_loss_exact',torch.equal(new,old))
    check('zero_shift_old_M5_feature_gradient_exact',torch.equal(dn,do))
    # The above seven images have uneven view counts: no platform-balanced reducer.
    rows=(zero.exp()*(zero-normalized_clip_logits(g.detach(),proj,text).div(2).log_softmax(-1))).sum(-1)
    check('KL_identity_sum_image_mean_Td_squared_once',torch.equal(new.detach(),4*rows.mean()))
    base=c.clone();base.defrost();base.ICLR_EXP.ENABLED=False;base.freeze()
    set_seed(1234,c.SOLVER.CUDNN_BENCHMARK);reference=make_model(base,500,7,2)
    reference_state=reference.state_dict();before_rng=torch.get_rng_state().clone()
    set_seed(1234,c.SOLVER.CUDNN_BENCHMARK);model=make_model(c,500,7,2)
    check('ICLR_disabled_original_student_state',set(reference_state)==set(model.state_dict()) and all(torch.equal(v,model.state_dict()[k]) for k,v in reference_state.items()))
    check('I2_init_preserves_CPU_RNG',torch.equal(before_rng,torch.get_rng_state()))
    check('no_TGVA_prompt_or_teacher_module',not hasattr(model,'iclr_tgva') and model.prompts is None and not any('teacher' in n for n,_ in model.named_modules()))
    check('frozen_original_P0',not model.base.clip_proj.requires_grad)
    check('zero_dense_C0',tuple(model.base.token_trajectory.gain.shape)==(11,1,129,768) and not torch.count_nonzero(model.base.token_trajectory.gain))
    del reference,reference_state;gc.collect()
    production_strict=True;strict_error=None
    with ExitStack() as stack:
        try:teacher,text,source,teacher_meta=load_resources(c,a.device)
        except ValueError as error:
            if not a.allow_local_position_roundoff or str(error)!='M2-5 teacher frozen original visual changed: pos_embed':raise
            if a.require_stats:raise ValueError('server preflight must use production strict teacher load') from error
            production_strict=False;strict_error=str(error)
            stack.enter_context(patch('utils.m5_checkpoint.load_teacher',local_position_diagnostic_loader))
            teacher,text,source,teacher_meta=load_resources(c,a.device)
        check('complete_teacher_saved_state_strict',all(not p.requires_grad for p in teacher.parameters()) and not teacher.training)
        all_rows,_=train_source(legacy_config(c));check('full_train_source_92133_500',len(all_rows)==92133 and len(source['pid_map'])==500)
        if a.require_stats:
            shift,stats=load_shift(c,source,teacher_meta,a.device)
        else:
            # Real clean train subset, with paired IDs in every modality. NEVER a formal cache.
            selected=[i for i in range(500) if source['condition_mask'][i].all()][0:2]
            tiny=[]
            for y in selected:
                for m in range(3):
                    for v in range(2):tiny.append(next(r for r in all_rows if r[1]==y and r[3]==m and int(r[2] in (5,6))==v))
            sums=torch.zeros(500,3,7,500,dtype=torch.float64);counts=torch.zeros(500,3,7,dtype=torch.long)
            loader=torch.utils.data.DataLoader(CleanTrain(tiny,c),batch_size=2)
            with torch.no_grad():
                for images,y,cam,m in loader:
                    z=teacher.forward_pre_bn(images.to(a.device),m.to(a.device),((cam==5)|(cam==6)).long().to(a.device))
                    ell=normalized_clip_logits(z,teacher.base.clip_proj,text).double().cpu();ell-=ell.mean(-1,keepdim=True)
                    for i in range(len(y)):sums[y[i],m[i],cam[i]]+=ell[i];counts[y[i],m[i],cam[i]]+=1
            raw_ids=[pid for pid,y in sorted(source['pid_map'].items(),key=lambda x:x[1])]
            stats=estimate(sums,counts,raw_ids);stats.update(complete=False,num_train_images=len(tiny),num_train_ids=2)
            shift=stats['shift_fp32'].to(a.device)
            check('real_clean_subset_estimator',torch.isfinite(shift).all(),dict(images=len(tiny),paired_ids=stats['num_paired_ids_by_modality']))
            partial=output/'partial_statistics.tmp';torch.save(stats,partial)
            partial_c=c.clone();partial_c.defrost();partial_c.ICLR_EXP.ASSETS.NEUTRAL_STATS=str(partial);partial_c.freeze()
            try:load_shift(partial_c,source,teacher_meta,a.device)
            except (ValueError,KeyError):check('partial_stats_rejected_by_production_loader',True)
            else:check('partial_stats_rejected_by_production_loader',False)
            partial.unlink()
        model.to(a.device).train();device=torch.device(a.device)
        pair=next(i for i in range(500) if source['condition_mask'][i].all())
        other=next(i for i in range(500) if i!=pair and source['condition_mask'][i,:,0].all())
        records=[]
        for m in range(3):
            records.append([next(r for r in all_rows if r[1]==other and r[3]==m and r[2] not in (5,6)),next(r for r in all_rows if r[1]==pair and r[3]==m and r[2] in (5,6))])
        inputs=[torch.stack([CleanTrain(group,c)[i][0] for i in range(2)]).to(device) for group in records]
        targets=torch.tensor([other,pair],device=device)
        if a.batch_per_modality==64:
            ids=[i for i in range(500) if source['condition_mask'][i,:,0].all()][:16]
            records=[[next(r for r in all_rows if r[1]==y and r[3]==m and r[2] not in (5,6)) for y in ids for _ in range(4)] for m in range(3)]
            inputs=[torch.stack([CleanTrain(group,c)[i][0] for i in range(64)]).to(device) for group in records]
            targets=torch.tensor(ids,device=device).repeat_interleave(4)
        cams=[torch.tensor([r[2] for r in group],device=device) for group in records];target_rep=targets.repeat(3)
        teacher_before={k:v.clone() for k,v in teacher.bottleneck.state_dict().items()}
        loss_fn,center=make_loss(c,500);optimizer,_=make_optimizer(c,model,center);scheduler=create_scheduler(c,optimizer,iters_per_epoch=469)
        check('P0_not_in_optimizer',all(p is not model.base.clip_proj for group in optimizer.param_groups for p in group['params']))
        roles=model.iclr_optimizer_roles
        check('optimizer_roles',all(r['lr']==5e-6 for r in roles if r['role']=='pretrained_visual') and all(r['lr']==3.5e-4 for r in roles if r['role'] in ('c0_gain','identity_head')))
        scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda');initial_p=model.base.clip_proj.detach().clone()
        skips=0;scheduler.step_update(100)
        for attempt in range(16):
            optimizer.zero_grad();bn_before=model.bottleneck.num_batches_tracked.clone();student_calls=[]
            h=model.base.register_forward_hook(lambda *_:student_calls.append(1))
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                out=model(inputs,camids=cams);aux=out[3]
                identity,ce,tri=loss_fn(out[0],out[1],target_rep)
                with torch.no_grad():
                    gt=teacher.forward_pre_bn(aux['images'],aux['mods'],aux['views'])
                    lp,lq=fixed_teacher_targets(normalized_clip_logits(gt,teacher.base.clip_proj,text),aux['mods'],aux['views'],shift)
                kd=neutral_kd(out[1],model.base.clip_proj,text,lq);total=identity+.5*kd
            h.remove()
            check('one_student_visual_forward',len(student_calls)==1)
            check('BN_once',int(model.bottleneck.num_batches_tracked-bn_before)==1)
            check('online_same_augmented_tensor',torch.equal(aux['images'],torch.cat(inputs)))
            check('semantic_FP32',lp.dtype==lq.dtype==kd.dtype==torch.float32)
            if attempt==0:
                head_grads=torch.autograd.grad(kd,[model.bottleneck.weight,model.classifier.weight],retain_graph=True,allow_unused=True)
                check('KD_bypasses_BN_classifier',all(x is None for x in head_grads))
                gain_grad=torch.autograd.grad(kd,model.base.token_trajectory.gain,retain_graph=True)[0]
                check('KD_gain_gradient_nonzero',gain_grad.double().norm()>0)
            scaler.scale(total).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=before:break
            skips+=1
        else:raise AssertionError('no finite update after default GradScaler calibration')
        check('finite_loss_and_gradient',bool(torch.isfinite(total) and torch.isfinite(norm)))
        check('P0_no_grad_or_change',model.base.clip_proj.grad is None and torch.equal(initial_p,model.base.clip_proj))
        check('teacher_no_grad_or_BN_change',all(x.grad is None for x in teacher.parameters()) and all(torch.equal(v,teacher.bottleneck.state_dict()[k]) for k,v in teacher_before.items()))
        # Exercise production runtime hooks, using in-memory partial stats ONLY for local diagnostic.
        if not a.require_stats:
            stack.enter_context(patch('tools.iclr_exp.neutral_runtime.load_shift',return_value=(shift,stats)))
            stack.enter_context(patch('tools.iclr_exp.neutral_runtime.file_digest',side_effect=lambda path:'PREFLIGHT_PARTIAL_NOT_FORMAL' if str(path)==c.ICLR_EXP.ASSETS.NEUTRAL_STATS else file_digest(path)))
        stack.enter_context(patch('tools.iclr_exp.neutral_runtime.load_resources',return_value=(teacher,text,source,teacher_meta)))
        from types import SimpleNamespace
        class SmallLoader:
            dataset=SimpleNamespace(dataset={})
            def __len__(self):return 1
        with tempfile.TemporaryDirectory(prefix='resume_',dir=output) as directory:
            rc=c.clone();rc.defrost();rc.ICLR_EXP.AUDIT.ENABLED=False;rc.OUTPUT_DIR=directory;rc.freeze()
            run=I2Run(rc,model,SmallLoader(),optimizer,scheduler,scaler)
            run.manifest['preflight_only']=True
            aux['targets']=target_rep
            before_bn={k:v.clone() for k,v in model.bottleneck.state_dict().items()};saved_rng=rng_state()
            run.gradient_probe(1,0,identity,aux,loss_fn)
            check('fixed_probe_BN_RNG_unchanged',all(torch.equal(v,model.bottleneck.state_dict()[k]) for k,v in before_bn.items()) and torch.equal(saved_rng['torch'],torch.get_rng_state()) and all(torch.equal(u,v) for u,v in zip(saved_rng['cuda'],torch.cuda.get_rng_state_all())))
            run.batches=skips;run.skips=skips;run.step(1,49,before,norm,ce,tri,kd,total.detach(),out[0],target_rep)
            run.save(1,0.,skips+1)
            obj=torch.load(Path(directory)/'resume_latest.pth',map_location='cpu',weights_only=False)
            check('full_checkpoint_has_no_teacher_or_text',not any(k.startswith(('teacher.','text.','iclr_tgva.')) for k in obj['state_dict']))
            del obj;gc.collect()
            resume=rc.clone();resume.defrost();resume.ICLR_EXP.RESUME=str(Path(directory)/'resume_latest.pth');resume.freeze()
            restored=I2Run(resume,model,SmallLoader(),optimizer,scheduler,scaler)
            check('production_resume_counters_and_fixed_probe',restored.start_epoch==2 and restored.batches==skips+1 and restored.updates==1 and restored.skips==skips and torch.equal(restored.probe_batch['images'],run.probe_batch['images']))
            # Real panel, eval OFF only for diagnostics; does not enter train graph.
            from datasets.whu_mars import WHU_MARS
            dataset=WHU_MARS(root=c.DATASETS.ROOT_DIR,modalities=list(c.DATASETS.MODALITIES),verbose=False)
            ac=rc.clone();ac.defrost();ac.ICLR_EXP.AUDIT.ENABLED=True;ac.ICLR_EXP.AUDIT.PAIRED_ID_COUNT=1;ac.ICLR_EXP.AUDIT.GROUND_ONLY_ID_COUNT=1;ac.ICLR_EXP.AUDIT.BATCH_SIZE=2;ac.OUTPUT_DIR=str(output);ac.freeze()
            restored.c=ac;restored.panel=build_panel(ac,dataset.train)
            saved_rng=rng_state();bn={k:v.clone() for k,v in model.bottleneck.state_dict().items()};restored.audit(0)
            check('real_clean_audit_BN_RNG_restored',all(torch.equal(v,model.bottleneck.state_dict()[k]) for k,v in bn.items()) and torch.equal(saved_rng['torch'],torch.get_rng_state()) and model.training and not teacher.training)
            with isolated_eval(model),torch.no_grad():expected=model(inputs[0],mode=1).cpu()
            deploy=deployment_state(model)
            reference=make_model(base,500,7,2);reference.load_state_dict(deploy,strict=True);reference.to(device).eval()
            with torch.no_grad():actual=reference(inputs[0],mode=1).cpu()
            check('deployment_strict_reload_features_exact',torch.equal(expected,actual),float((expected-actual).abs().max()))
            torch.save(dict(state_dict=deploy,metadata=dict(implementation=NEUTRAL_VERSION,variant='m5_neutral',epoch=0,
                original_clip_sha256=file_digest(c.MODEL.PRETRAIN_PATH),student=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=[11,1,129,768]))),output/'preflight_student.pth')
        report=dict(status='passed',production_strict_teacher=production_strict,strict_teacher_error=strict_error,
            formal_statistics_checked=a.require_stats,torch_version=torch.__version__,device=a.device,
            images_per_batch=3*a.batch_per_modality,initial_amp_skips=skips,scaler_scale=scaler.get_scale(),
            scope='method checks + real old teacher/text + small actual train-image backward/reload; not formal training',checks=checks)
        (output/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print('I2_PREFLIGHT_OK',output/'preflight.json',flush=True)


if __name__=='__main__':main()
