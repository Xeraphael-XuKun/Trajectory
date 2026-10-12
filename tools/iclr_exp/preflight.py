"""Necessary I1 checks on real CLIP plus a small synthetic training batch."""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description='I1 本地/服务器必要预检，不启动正式训练')
    p.add_argument('--resources', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-per-modality',type=int,default=2,help='本地2；服务器64核验正式192图逻辑batch的显存与前反向')
    a = p.parse_args()
    if a.batch_per_modality not in (2,64):
        p.error('--batch-per-modality must be 2 (local) or 64 (formal batch shape)')
    import torch
    import torch.nn.functional as F
    from tools.iclr_exp.configuration import load_config
    from tools.iclr_exp.runtime import deployment_state
    from tools.iclr_exp.audit import isolated_eval, build_panel, run_audit
    from tools.m2.artifacts import read_clip, rng_state, restore_rng
    from loss.iclr_exp import tgva_loss, tgva_from_aux
    from model import make_model
    from loss import make_loss
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from train import set_seed
    torch.set_num_threads(a.threads)
    c = load_config(resources=a.resources)
    checks = {}
    def check(name, ok, evidence=None):
        if not ok:
            raise AssertionError(name + ': ' + str(evidence))
        checks[name] = evidence if evidence is not None else True
    mods = torch.tensor([0,0,1,1,2,2]); views = torch.tensor([0,1,0,1,0,1])
    uniform = torch.zeros(6,6, requires_grad=True)
    check('uniform_two_platform_reduction', abs(float(tgva_loss(uniform,uniform,mods,views))-4*math.log(6))<1e-6)
    check('uniform_one_platform_reduction', abs(float(tgva_loss(uniform,uniform,mods,views*0))-2*math.log(6))<1e-6)
    check('labels_ground_zero_aerial_one', (2*mods+views).tolist()==[0,1,2,3,4,5])
    from tools.iclr_exp.eval_and_export import ranking_rows
    gen_rank=torch.Generator().manual_seed(20261012)
    qmeta=[dict(raw_pid=y,camera_zero_based=i%7,relative_path='q%d'%i,modality=i%3,platform=0) for i,y in enumerate([0,1,99])]
    gmeta=[dict(raw_pid=i%6,camera_zero_based=i%7,relative_path='g%d'%i,modality=i%3,platform=int(i%7 in (5,6))) for i in range(100)]
    qr=F.normalize(torch.randn(3,8,generator=gen_rank),dim=-1)
    gr=F.normalize(torch.randn(100,8,generator=gen_rank),dim=-1)
    rr=ranking_rows(qr,gr,qmeta,gmeta,q_chunk=2)
    check('ranking_matches_original_sysu_AP_INP_R1',rr[0]['valid'] and rr[1]['valid'] and not rr[2]['valid'] and rr[2]['AP'] is None)
    check('same_camera_negatives_removed',all(r['top1']['camera_zero_based']!=r['camera_zero_based'] for r in rr))
    base_c = c.clone(); base_c.defrost(); base_c.ICLR_EXP.ENABLED=False; base_c.freeze()
    set_seed(1234,c.SOLVER.CUDNN_BENCHMARK)
    reference = make_model(base_c,500,7,2)
    check('disabled_no_auxiliary', not hasattr(reference,'iclr_tgva') and reference.prompts is None)
    state_rng = torch.random.get_rng_state().clone()
    set_seed(1234,c.SOLVER.CUDNN_BENCHMARK)
    model = make_model(c,500,7,2)
    check('auxiliary_initialization_preserves_cpu_rng', torch.equal(state_rng,torch.random.get_rng_state()))
    check('original_student_initial_state', all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items()))
    clip = read_clip(c.MODEL.PRETRAIN_PATH)
    check('projection_original_exact', torch.equal(model.base.clip_proj.detach(),clip['visual.proj'].float()))
    check('logit_scale_original_exact', torch.equal(model.iclr_tgva.logit_scale,clip['logit_scale'].float().exp()))
    check('zero_C0_gain_shape', tuple(model.base.token_trajectory.gain.shape)==(11,1,129,768) and not torch.count_nonzero(model.base.token_trajectory.gain))
    check('learnable_prompt_count', sum(x.numel() for x in model.iclr_tgva.parameters() if x.requires_grad)==4608)
    check('tower_frozen', all(not x.requires_grad for x in model.iclr_tgva.prompts.text.parameters()))
    del clip
    device = torch.device(a.device)
    model.to(device)
    reference.to(device)
    gen = torch.Generator().manual_seed(20261012)
    batch=a.batch_per_modality
    images = [torch.randn(batch,3,256,128,generator=gen).to(device) for _ in range(3)]
    cameras = [torch.tensor([0,5]*(batch//2),device=device) for _ in range(3)]
    x = torch.cat(images)
    model.eval(); reference.eval()
    with torch.no_grad():
        g_on = model.base(x); g_off = model.base(x,trajectory_gate=x.new_zeros(len(x)))
        check('zero_gain_eval_on_off_exact', torch.equal(g_on,g_off))
        check('disabled_forward_matches', torch.equal(reference.base(x),g_on))
        model.base.token_trajectory.gain.fill_(.001)
        corrections = []
        hook = model.base.token_trajectory.register_forward_hook(lambda _,args,result: corrections.append((int(args[0]),bool(torch.count_nonzero(result)))))
        off = model.base(x,trajectory_gate=x.new_zeros(len(x)))
        hook.remove()
        check('all_11_injections_disabled', corrections==[(i,False) for i in range(1,12)],corrections)
        on = model.base(x)
        check('nonzero_gain_changes_on_only', torch.equal(off,g_off) and not torch.allclose(on,off,atol=1e-6))
        model.base.token_trajectory.gain.zero_()
    # Same small batch, actual three-modal identity CE/Triplet and two live graphs.
    model.train()
    bn_before = model.bottleneck.num_batches_tracked.clone()
    records = []
    def begin(_, args):
        records.append({'before':torch.random.get_rng_state().clone(),
                        'cuda_before':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
    def end(_, args, out):
        records[-1]['after']=torch.random.get_rng_state().clone()
        records[-1]['cuda_after']=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    h1=model.base.register_forward_pre_hook(begin); h2=model.base.register_forward_hook(end)
    targets=(torch.tensor([0,1],device=device) if batch==2 else torch.arange(16,device=device).repeat_interleave(4)).repeat(3)
    amp_enabled=device.type=='cuda'
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        out=model(images,camids=cameras)
        loss_fn, center=make_loss(c,500)
        identity, ce, tri=loss_fn(out[0],out[1],targets)
    h1.remove(); h2.remove()
    raw, h_on,h_off=tgva_from_aux(out[3])
    check('semantic_logits_FP32',h_on.dtype==torch.float32 and h_off.dtype==torch.float32 and raw.dtype==torch.float32)
    check('BN_once',int(model.bottleneck.num_batches_tracked-bn_before)==1)
    check('prompt_encoder_once_per_step',out[3]['text'].shape==(6,512))
    key='cuda_' if amp_enabled else ''
    check('independent_rng_continuation',torch.equal(records[1][key+'before'],records[0][key+'after']) and not torch.equal(records[1][key+'after'],records[1][key+'before']))
    from loss.iclr_exp import tgva_loss
    off_loss=F.cross_entropy(h_off,2*out[3]['mods']+out[3]['views'])
    gain=model.base.token_trajectory.gain
    qkv=model.base.blocks[-1].attn.qkv.weight
    grad_off=torch.autograd.grad(off_loss,[gain,qkv],retain_graph=True,allow_unused=True)
    check('off_no_gain_gradient',grad_off[0] is None or not torch.count_nonzero(grad_off[0]))
    check('off_live_backbone_gradient',grad_off[1] is not None and grad_off[1].double().norm()>0)
    optimizer,_=make_optimizer(c,model,center)
    scheduler=create_scheduler(c,optimizer,iters_per_epoch=457)
    roles=model.iclr_optimizer_roles
    check('optimizer_roles',next(r for r in roles if r['name']=='base.clip_proj')['lr']==5e-6 and all(r['lr']==3.5e-4 for r in roles if r['role'] in ('c0_gain','condition_prompt','identity_head')))
    check('optimizer_unique',len({id(x) for g in optimizer.param_groups for x in g['params']})==len(optimizer.param_groups))
    scaler=torch.cuda.amp.GradScaler(enabled=amp_enabled)
    projection_before=model.base.clip_proj.detach().clone()
    prompt_before=model.iclr_tgva.prompts.shared_ctx.detach().clone()
    # Default GradScaler may legitimately skip early steps while reducing scale.
    # Exercise that real behavior rather than silently lowering production scale.
    initial_amp_skips=0
    scheduler.step_update(100)
    for attempt in range(16):
        if attempt:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out=model(images,camids=cameras)
                identity,ce,tri=loss_fn(out[0],out[1],targets)
            raw,h_on,h_off=tgva_from_aux(out[3])
        total=identity+5*raw
        scaler.scale(total).backward(); scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scale_before=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()>=scale_before:
            break
        initial_amp_skips+=1
        check('AMP_skip_preserves_parameters_attempt_%d'%attempt,
            torch.equal(projection_before,model.base.clip_proj) and torch.equal(prompt_before,model.iclr_tgva.prompts.shared_ctx))
    else:
        raise AssertionError('no finite CUDA update after GradScaler calibration')
    check('on_gain_gradient',gain.grad is not None and gain.grad.double().norm()>0)
    check('projection_gradient',model.base.clip_proj.grad is not None and model.base.clip_proj.grad.double().norm()>0)
    check('prompt_gradient',model.iclr_tgva.prompts.shared_ctx.grad is not None and model.iclr_tgva.prompts.shared_ctx.grad.double().norm()>0)
    check('text_tower_no_grad',all(x.grad is None for x in model.iclr_tgva.prompts.text.parameters()))
    check('finite_total_gradient',bool(torch.isfinite(total) and torch.isfinite(grad_norm)))
    check('small_step_not_skipped',scaler.get_scale()>=scale_before)
    check('projection_and_prompt_updated',not torch.equal(projection_before,model.base.clip_proj) and not torch.equal(prompt_before,model.iclr_tgva.prompts.shared_ctx))
    # Strict deployment reload into the original C0 (no prompt/text architecture).
    reference.load_state_dict({k:v.cpu() for k,v in deployment_state(model).items()},strict=True)
    reference.to(device)
    with isolated_eval(model),torch.no_grad():
        expected=model(x,mode=1).cpu()
        actual=reference(x,mode=1).cpu()
        check('deployment_reload_features',torch.equal(expected,actual),float((expected-actual).abs().max()))
    # Full checkpoint includes optimizer/scaler/scheduler/RNG, strict student state.
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    temporary=output/'preflight_reload.tmp'
    saved_rng=rng_state()
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=saved_rng),temporary)
    saved=torch.load(temporary,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
    check('full_checkpoint_reload',torch.equal(saved_rng['torch'],torch.random.get_rng_state()) if 'torch' in saved_rng else True)
    # Exercise the actual production resume hooks, not only torch serialization.
    import tempfile
    from types import SimpleNamespace
    from tools.iclr_exp.runtime import I1Run
    class SmallLoader:
        dataset=SimpleNamespace(dataset={})
        def __len__(self):return 1
    with tempfile.TemporaryDirectory(prefix='runtime_resume_',dir=str(output)) as runtime_dir:
        rc=c.clone(); rc.defrost(); rc.ICLR_EXP.AUDIT.ENABLED=False; rc.OUTPUT_DIR=runtime_dir; rc.freeze()
        run=I1Run(rc,model,SmallLoader(),optimizer,scheduler,scaler)
        run.manifest['preflight_synthetic']=True
        run.batches=initial_amp_skips; run.skips=initial_amp_skips
        run.step(1,49,scaler.get_scale(),grad_norm,ce,tri,raw,total.detach(),out[0],targets)
        moments={k:v.detach().clone() for k,v in optimizer.state[model.base.clip_proj].items() if torch.is_tensor(v)}
        run.save(1,0.)
        resume_c=rc.clone(); resume_c.defrost(); resume_c.ICLR_EXP.RESUME=str(Path(runtime_dir)/'resume_latest.pth'); resume_c.freeze()
        restored=I1Run(resume_c,model,SmallLoader(),optimizer,scheduler,scaler)
        check('production_resume_counters',restored.start_epoch==2 and restored.batches==initial_amp_skips+1 and restored.updates==1 and restored.skips==initial_amp_skips)
        check('production_resume_optimizer_moments',all(torch.equal(v,optimizer.state[model.base.clip_proj][k]) for k,v in moments.items()))
    from tools.iclr_exp.runtime import VERSION
    from tools.m2.artifacts import file_digest
    torch.save(dict(state_dict=deployment_state(model),metadata=dict(implementation=VERSION,
        variant='tgva_ref',epoch=0,original_clip_sha256=file_digest(c.MODEL.PRETRAIN_PATH),
        student=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=[11,1,129,768]))),
        output/'preflight_student.pth')
    temporary.unlink()
    del saved,optimizer,center,out,identity,total,raw
    # Exercise a real clean training panel without using any query/gallery labels.
    from datasets.whu_mars import WHU_MARS
    dataset=WHU_MARS(root=c.DATASETS.ROOT_DIR,modalities=list(c.DATASETS.MODALITIES),verbose=False)
    panel_c=c.clone(); panel_c.defrost(); panel_c.ICLR_EXP.AUDIT.PAIRED_ID_COUNT=1; panel_c.ICLR_EXP.AUDIT.GROUND_ONLY_ID_COUNT=1; panel_c.ICLR_EXP.AUDIT.BATCH_SIZE=2; panel_c.OUTPUT_DIR=str(output)
    panel=build_panel(panel_c,dataset.train)
    before_bn={k:v.clone() for k,v in model.bottleneck.state_dict().items()}
    rng_before=rng_state()
    with isolated_eval(model),torch.no_grad():
        initial_anchors=model.iclr_tgva().cpu()
    run_audit(panel_c,model,panel,0,projection_before.cpu(),initial_anchors)
    check('audit_BN_unchanged',all(torch.equal(v,model.bottleneck.state_dict()[k]) for k,v in before_bn.items()))
    check('audit_rng_unchanged',torch.equal(rng_before['torch'],torch.random.get_rng_state()) if 'torch' in rng_before else True)
    check('audit_restores_training_and_frozen_tower',model.training and not model.iclr_tgva.prompts.text.training)
    report=dict(status='passed',device=str(device),visual_amp=amp_enabled,torch_version=torch.__version__,
        scope='synthetic %d-image batch plus real clean training audit panel; not 60-epoch training'%(3*batch),
        logical_batch_images=3*batch,
        checks=checks,initial_amp_skips=initial_amp_skips,final_scaler_scale=scaler.get_scale(),synthetic_loss=dict(ce=float(ce.detach()),triplet=float(tri.detach())),
        prompt_metadata=model.iclr_tgva.metadata(),train_images=dataset.num_train_imgs,train_ids=dataset.num_train_pids)
    (output/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('I1_PREFLIGHT_OK: '+str(output/'preflight.json'))


if __name__=='__main__':
    main()
