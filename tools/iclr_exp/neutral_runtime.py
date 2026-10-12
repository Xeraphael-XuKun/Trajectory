"""One student pass + frozen online old teacher; no teacher in student state."""
import json
import os
import time
from pathlib import Path
import torch
from tools.iclr_exp.runtime import I1Run, NEUTRAL_VERSION, git
from tools.iclr_exp.audit import build_panel, isolated_eval
from tools.iclr_exp.neutral_assets import load_resources, load_shift, fingerprint, paired_mask
from tools.m2.artifacts import file_digest
from loss.iclr_neutral import fixed_teacher_targets, neutral_kd, target_quality
from model.m2_teacher import normalized_clip_logits


class I2Run(I1Run):
    implementation = NEUTRAL_VERSION
    variant = 'm5_neutral'
    target_version = 'ellT+(.5-view)*shift; tau0=.07; Td=2; lambda=.5'
    loss_weight = .5
    resume_keys = ('implementation','variant','original_clip_sha256','student','resources','neutral_stats_sha256','target_version')

    def __init__(self,c,model,train_loader,optimizer,scheduler,scaler):
        self.c,self.model=c,model
        self.optimizer,self.scheduler,self.scaler=optimizer,scheduler,scaler
        self.batches=self.updates=self.skips=0; self.start_epoch=1
        self.epoch_train_seconds=self.eval_seconds=self.audit_seconds=self.previous_wall=0.
        self.wall_start=time.time()
        device=next(model.parameters()).device
        self.teacher,self.text,source,meta=load_resources(c,device)
        self.shift,self.stats=load_shift(c,source,meta,device)
        if not torch.equal(self.teacher.base.clip_proj,model.base.clip_proj):
            raise ValueError('student P0 differs from original frozen teacher P0')
        self.paired=paired_mask(self.stats,device)
        self.manifest=dict(implementation=self.implementation,variant=self.variant,
            source_base_commit='db8a5c65ba0b7b5f0c62d00d2adb2f5e3bfe1177',
            infrastructure_commit='b0bcad8631d6cc0853809a4db319f34421331ee4',
            code_commit=git(['rev-parse','HEAD']).strip(),dirty_diff=git(['diff','HEAD']),
            untracked_sources=git(['ls-files','--others','--exclude-standard']).splitlines(),
            config=c.dump(),original_clip_sha256=source['clip_sha256'],
            resources=fingerprint(c,meta,source),neutral_stats_sha256=file_digest(c.ICLR_EXP.ASSETS.NEUTRAL_STATS),
            target_version=self.target_version,pair_rng=c.ICLR_EXP.PAIR_RNG,
            teacher_used=True,prompt_stage_used=False,projection_trainable=False,
            student=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=[11,1,129,768]),
            nominal_loader_batches=len(train_loader),logical_images_per_batch=192,
            online_visual_precision='original M5 autocast; semantic FP32',
            offline_visual_precision='FP32; FP64 sums',
            teacher_architecture=meta['teacher_architecture'], teacher_epoch=meta['epoch'],
            paired_ids_by_modality=self.stats['num_paired_ids_by_modality'])
        self.initial_projection=model.base.clip_proj.detach().float().cpu().clone()
        self.initial_anchors=None
        self.probe_batch=None
        self.panel=build_panel(c,train_loader.dataset.dataset) if c.ICLR_EXP.AUDIT.ENABLED else None
        if c.ICLR_EXP.RESUME:self.restore(c.ICLR_EXP.RESUME)
        elif (Path(c.OUTPUT_DIR)/'resume_latest.pth').exists():
            raise ValueError('output contains checkpoint; use --resume or new output')
        self.write('run_manifest.json',self.manifest)
        self.write('neutral_stats_summary.json',{k:self.stats[k] for k in ('num_paired_ids_by_modality','crossfit_fold_ids','crossfit_summary','fit_residual_mean_norm')})
        if self.panel is not None:
            self.write('audit_panel.json',self.panel)
            if self.start_epoch==1:self.audit(0)

    def loss(self,aux,targets=None):
        if aux['kind']!='iclr_neutral':raise ValueError('I2 received another method auxiliary')
        self.teacher.eval()
        with torch.no_grad():
            g=self.teacher.forward_pre_bn(aux['images'],aux['mods'],aux['views'])
            logits=normalized_clip_logits(g,self.teacher.base.clip_proj,self.text,.07)
            aux['log_p'],aux['log_q']=fixed_teacher_targets(logits,aux['mods'],aux['views'],self.shift)
        return neutral_kd(aux['g_on'],aux['projection'],self.text,aux['log_q'])

    def step(self,epoch,iteration,scale_before,grad_norm,identity,triplet,kd,total,score,targets):
        self.batches+=1
        success=self.scaler.get_scale()>=scale_before
        self.updates+=int(success);self.skips+=int(not success)
        if (iteration+1)%self.c.SOLVER.LOG_PERIOD:return
        norm=float(grad_norm);finite=bool(torch.isfinite(grad_norm))
        row=dict(epoch=epoch,epoch_batch=iteration+1,observed_batches=self.batches,
            successful_updates=self.updates,amp_skips=self.skips,ce=float(identity.detach()),
            triplet=float(triplet.detach()),raw_kl=float(kd.detach()/4),temperature_squared_kd=float(kd.detach()),
            weighted_kd=float(.5*kd.detach()),total_loss=float(total),scaler_scale=float(self.scaler.get_scale()),
            global_grad_norm_before_clip=norm if finite else None,
            clip_coefficient=min(1.,1./(norm+1e-6)) if finite else None,
            classifier_accuracy=float((score.detach().argmax(-1)==targets).float().mean()),
            parameter_group_lrs=[g['lr'] for g in self.optimizer.param_groups])
        with open(os.path.join(self.c.OUTPUT_DIR,'training_terms.jsonl'),'a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def gradient_probe(self,epoch,iteration,identity_loss,aux,loss_fn=None):
        if iteration!=0 or epoch not in (1,10,30,60):return
        # Persist the first augmented real PKM tensor, reuse it across all epochs.
        if self.probe_batch is None:
            self.probe_batch=dict(images=aux['images'].detach().cpu(),
                mods=aux['mods'].cpu(),views=aux['views'].cpu(),targets=aux['targets'].cpu())
        device=next(self.model.parameters()).device
        batch={k:v.to(device) for k,v in self.probe_batch.items()}
        x=batch['images']
        # Eval probe keeps BN/RNG untouched and fixes DropPath; main train remains train+AMP.
        with isolated_eval(self.model),torch.cuda.amp.autocast(enabled=True):
            self.model.train(False)
            g=self.model.base(x);f=self.model.bottleneck(g);score=self.model.classifier(f)
            identity=loss_fn(score,g,batch['targets'])[0]
            probe=dict(kind='iclr_neutral',g_on=g,images=x,mods=batch['mods'],views=batch['views'],projection=self.model.base.clip_proj)
            kd=self.loss(probe)
            names=['base.token_trajectory.gain','base.blocks.11.attn.qkv.weight','base.norm.weight','bottleneck.weight','classifier.weight']
            params=dict(self.model.named_parameters());selected=[params[name] for name in names]
            a=torch.autograd.grad(identity,selected,retain_graph=True,allow_unused=True)
            b=torch.autograd.grad(.5*kd,selected,allow_unused=True)
            rows=[]
            for name,u,v in zip(names,a,b):
                def desc(t):return dict(status='none' if t is None else ('zero' if not torch.count_nonzero(t) else 'nonzero'),norm=None if t is None else float(t.detach().double().norm()))
                cosine=None
                if u is not None and v is not None and u.double().norm()>0 and v.double().norm()>0:
                    cosine=float((u.double().flatten()@v.double().flatten())/(u.double().norm()*v.double().norm()))
                rows.append(dict(parameter=name,identity=desc(u),weighted_kd=desc(v),cosine=cosine))
            quality=target_quality(probe['log_p'],probe['log_q'],batch['targets'],batch['mods'],batch['views'],self.paired[batch['targets'],batch['mods']],normalized_clip_logits(g.detach(),self.model.base.clip_proj,self.text).div(2).log_softmax(-1))
        self.write('gradient_epoch%d.json'%epoch,dict(input='fixed first real augmented PKM batch; eval BN/DropPath; visual AMP',
            frozen_projection=True,rows=rows))
        self.write('target_augmented_epoch%d.json'%epoch,quality)

    def audit(self,epoch):
        if self.panel is not None and epoch in self.c.ICLR_EXP.AUDIT.EPOCHS:
            from tools.iclr_exp.neutral_audit import run_audit
            start=time.time();run_audit(self,epoch);self.audit_seconds+=time.time()-start
