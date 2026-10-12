"""I3: online teacher first, matched student graphs, separately audited gradients."""
import json
import os
from pathlib import Path
import torch
from tools.iclr_exp.neutral_runtime import I2Run
from tools.iclr_exp.runtime import PAIRED_VERSION
from tools.iclr_exp.audit import isolated_eval
from model.iclr_exp.paired_forward import paired_visual_forward
from model.m2_teacher import normalized_clip_logits
from loss.iclr_neutral import fixed_teacher_targets,target_quality
from loss.iclr_paired import paired_terms,branch_quality


class I3Run(I2Run):
    implementation=PAIRED_VERSION
    variant='m5_paired'
    target_version='OFF:pT; ON:qT=ellT+(.5-view)*shift; tau0=.07; Td=2; branch_mean; lambda=.5; replay'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.manifest.update(infrastructure_commit='7941f1b56dcb6e2c09ba233546b42b0ed034ff5b',
            student_visual_passes_per_main_batch=2,teacher_visual_passes_per_main_batch=1,
            teacher_optimizer_updates=0,id_prompt_optimizer_updates=0)
        self.write('run_manifest.json',self.manifest)
        counts=self.stats['group_counts'].cpu();d=self.stats['shift_fp64'].double().cpu()
        summary={k:self.stats[k] for k in ('num_paired_ids_by_modality','crossfit_fold_ids','crossfit_summary','fit_residual_mean_norm')}
        summary.update(shared_cache_sha256=self.manifest['neutral_stats_sha256'],
            complete=self.stats['complete'],images_by_modality_camera=counts.sum(0).tolist(),
            IDs_by_modality_camera=(counts>0).sum(0).tolist(),
            shift_norm=d.norm(dim=-1).tolist(),shift_maxabs=d.abs().amax(-1).tolist(),
            shift_class_mean=d.mean(-1).tolist(),paired_identity_gap_norms=[self.stats['D_by_identity'][self.stats['paired_ids_by_modality'][m],m].norm(dim=-1).tolist() for m in range(3)])
        self.write('neutral_stats_summary.json',summary)

    @torch.no_grad()
    def prepare(self,images,camids):
        self.teacher.eval()
        x=torch.cat(images)
        mods=torch.arange(3,device=x.device).repeat_interleave(len(images[0]))
        cams=torch.cat(camids);views=((cams==5)|(cams==6)).long()
        g=self.teacher.forward_pre_bn(x,mods,views)
        logp,logq=fixed_teacher_targets(normalized_clip_logits(g,self.teacher.base.clip_proj,self.text),mods,views,self.shift)
        return dict(log_p=logp,log_q=logq)

    def loss(self,aux,targets=None):
        if aux['kind']!='iclr_paired':raise ValueError('I3 received another method auxiliary')
        kd,off,on=paired_terms(aux['g_on'],aux['g_off'],aux['projection'],self.text,aux['log_p'],aux['log_q'])
        self.last_terms=dict(raw_off=off.detach(),raw_on=on.detach())
        return kd

    def step(self,epoch,iteration,scale_before,grad_norm,identity,triplet,kd,total,score,targets):
        self.batches+=1
        success=self.scaler.get_scale()>=scale_before
        self.updates+=int(success);self.skips+=int(not success)
        if (iteration+1)%self.c.SOLVER.LOG_PERIOD:return
        norm=float(grad_norm);finite=bool(torch.isfinite(grad_norm))
        row=dict(epoch=epoch,epoch_batch=iteration+1,observed_batches=self.batches,
            successful_updates=self.updates,amp_skips=self.skips,ce=float(identity.detach()),
            triplet=float(triplet.detach()),raw_off_kl=float(self.last_terms['raw_off']),raw_on_kl=float(self.last_terms['raw_on']),
            raw_branch_mean=float(kd.detach()/4),temperature_squared_kd=float(kd.detach()),
            weighted_kd=float(.5*kd.detach()),total_loss=float(total),scaler_scale=float(self.scaler.get_scale()),
            global_grad_norm_before_clip=norm if finite else None,
            clip_coefficient=min(1.,1./(norm+1e-6)) if finite else None,
            classifier_accuracy=float((score.detach().argmax(-1)==targets).float().mean()),
            parameter_group_lrs=[g['lr'] for g in self.optimizer.param_groups])
        with open(os.path.join(self.c.OUTPUT_DIR,'training_terms.jsonl'),'a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def gradient_probe(self,epoch,iteration,identity_loss,aux,loss_fn=None):
        if iteration!=0 or epoch not in (1,10,30,60):return
        if self.probe_batch is None:
            self.probe_batch=dict(images=aux['images'].detach().cpu(),mods=aux['mods'].cpu(),
                views=aux['views'].cpu(),targets=aux['targets'].cpu())
        device=next(self.model.parameters()).device
        batch={k:v.to(device) for k,v in self.probe_batch.items()};x=batch['images']
        with isolated_eval(self.model),torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                gt=self.teacher.forward_pre_bn(x,batch['mods'],batch['views'])
                lp,lq=fixed_teacher_targets(normalized_clip_logits(gt,self.teacher.base.clip_proj,self.text),batch['mods'],batch['views'],self.shift)
            on,off=paired_visual_forward(self.model.base,x)
            identity=loss_fn(self.model.classifier(self.model.bottleneck(on)),on,batch['targets'])[0]
            kd,raw_off,raw_on=paired_terms(on,off,self.model.base.clip_proj,self.text,lp,lq)
            names=['base.token_trajectory.gain','base.blocks.11.attn.qkv.weight','base.norm.weight','bottleneck.weight','classifier.weight']
            selected=[dict(self.model.named_parameters())[name] for name in names]
            # Expanded I3 weights: each raw KL has weight 1 (0.5 * 4 * 0.5).
            gradients={name:torch.autograd.grad(loss,selected,retain_graph=True,allow_unused=True)
                for name,loss in [('identity',identity),('weighted_off',raw_off),('weighted_on',raw_on)]}
            rows=[]
            def desc(g):
                return dict(status='none' if g is None else ('zero' if not torch.count_nonzero(g) else 'nonzero'),
                    norm=None if g is None else float(g.detach().double().norm()))
            for i,name in enumerate(names):
                row=dict(parameter=name,**{task:desc(gs[i]) for task,gs in gradients.items()},cosines={})
                for a,b in [('identity','weighted_off'),('identity','weighted_on'),('weighted_off','weighted_on')]:
                    u,v=gradients[a][i],gradients[b][i]
                    row['cosines'][a+'_vs_'+b]=float((u.double().flatten()@v.double().flatten())/(u.double().norm()*v.double().norm())) if u is not None and v is not None and u.double().norm()>0 and v.double().norm()>0 else None
                rows.append(row)
            lon=normalized_clip_logits(on.detach(),self.model.base.clip_proj,self.text).div(2).log_softmax(-1)
            loff=normalized_clip_logits(off.detach(),self.model.base.clip_proj,self.text).div(2).log_softmax(-1)
            coverage=self.paired[batch['targets'],batch['mods']]
            quality=dict(on=target_quality(lp,lq,batch['targets'],batch['mods'],batch['views'],coverage,lon),
                off=target_quality(lp,lq,batch['targets'],batch['mods'],batch['views'],coverage,loff),
                branches=branch_quality(lon,loff,lp,lq,batch['mods'],batch['views'],self.shift))
        self.write('gradient_epoch%d.json'%epoch,dict(input='fixed first augmented PKM batch; eval BN/DropPath; visual AMP',
            weights='each raw off/on KL=1; identity=1',rows=rows))
        self.write('target_augmented_epoch%d.json'%epoch,quality)
