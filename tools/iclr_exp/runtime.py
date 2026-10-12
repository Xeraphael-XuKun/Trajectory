"""Logging, epoch-boundary resume and deployment hooks for the original loop."""
import json
import os
import subprocess
import time
from pathlib import Path
import torch
from tools.m2.artifacts import file_digest, rng_state, restore_rng
from tools.iclr_exp.audit import build_panel, run_audit, isolated_eval

VERSION = 'iclr-i1-tgva-ref-v1'
ROOT = Path(__file__).resolve().parents[2]


def git(args):
    return subprocess.check_output(['git']+args, cwd=str(ROOT), text=True, encoding='utf-8')


def deployment_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('iclr_tgva.')}


class I1Run:
    def __init__(self, c, model, train_loader, optimizer, scheduler, scaler):
        self.c, self.model = c, model
        self.optimizer, self.scheduler, self.scaler = optimizer, scheduler, scaler
        self.batches = self.updates = self.skips = 0
        self.start_epoch = 1
        self.epoch_train_seconds = self.eval_seconds = self.audit_seconds = 0.
        self.previous_wall = 0.
        self.wall_start = time.time()
        self.manifest = dict(implementation=VERSION, variant='tgva_ref',
            source_base_commit='db8a5c65ba0b7b5f0c62d00d2adb2f5e3bfe1177',
            code_commit=git(['rev-parse', 'HEAD']).strip(), dirty_diff=git(['diff', 'HEAD']),
            untracked_sources=git(['ls-files', '--others', '--exclude-standard']).splitlines(),
            config=c.dump(), original_clip_sha256=file_digest(c.ICLR_EXP.ASSETS.ORIGINAL_CLIP),
            teacher_used=False, prompt_stage_used=False, projection_trainable=True,
            student=dict(trajectory_variant='dense', acceleration_mix=0., gain_shape=[11,1,129,768]),
            nominal_loader_batches=len(train_loader), logical_images_per_batch=192,
            prompt=model.iclr_tgva.metadata())
        self.initial_projection = model.base.clip_proj.detach().float().cpu().clone()
        with isolated_eval(model), torch.no_grad():
            self.initial_anchors = model.iclr_tgva().cpu()
        self.panel = build_panel(c, train_loader.dataset.dataset) if c.ICLR_EXP.AUDIT.ENABLED else None
        if c.ICLR_EXP.RESUME:
            self.restore(c.ICLR_EXP.RESUME)
        else:
            # Prevent accidental overwrite when the launcher is rerun.
            if (Path(c.OUTPUT_DIR)/'resume_latest.pth').exists():
                raise ValueError('output already contains a training checkpoint; use --resume or a new output directory')
        self.write('run_manifest.json', self.manifest)
        if self.panel is not None:
            self.write('audit_panel.json', self.panel)
            if self.start_epoch == 1:
                self.audit(0)

    def write(self, name, value):
        with open(os.path.join(self.c.OUTPUT_DIR, name), 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

    def audit(self, epoch):
        if self.panel is not None and epoch in self.c.ICLR_EXP.AUDIT.EPOCHS:
            start = time.time()
            run_audit(self.c, self.model, self.panel, epoch, self.initial_projection, self.initial_anchors)
            self.audit_seconds += time.time()-start

    def step(self, epoch, iteration, scale_before, grad_norm, identity, triplet, tgva, total, score, targets):
        self.batches += 1
        success = self.scaler.get_scale() >= scale_before
        self.updates += int(success); self.skips += int(not success)
        if (iteration+1) % self.c.SOLVER.LOG_PERIOD:
            return
        norm = float(grad_norm)
        row = dict(epoch=epoch, epoch_batch=iteration+1, observed_batches=self.batches,
            successful_updates=self.updates, amp_skips=self.skips,
            ce=float(identity.detach()), triplet=float(triplet.detach()),
            raw_tgva=float(tgva.detach()), weighted_tgva=float(5*tgva.detach()),
            total_loss=float(total), scaler_scale=float(self.scaler.get_scale()),
            global_grad_norm_before_clip=norm if torch.isfinite(grad_norm) else None,
            clip_coefficient=min(1., 1./(norm+1e-6)) if torch.isfinite(grad_norm) else None,
            classifier_accuracy=float((score.detach().argmax(-1)==targets).float().mean()),
            parameter_group_lrs=[g['lr'] for g in self.optimizer.param_groups])
        with open(os.path.join(self.c.OUTPUT_DIR, 'training_terms.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')

    def gradient_probe(self, epoch, iteration, identity_loss, aux):
        if iteration != 0 or epoch not in (1,10,30,60):
            return
        from loss.iclr_exp import tgva_from_aux
        raw, h_on, h_off = tgva_from_aux(aux)
        params = dict(self.model.named_parameters())
        names = ['base.token_trajectory.gain', 'base.blocks.11.attn.qkv.weight',
                 'base.norm.weight', 'base.clip_proj', 'iclr_tgva.prompts.shared_ctx']
        selected = [params[n] for n in names]
        a = torch.autograd.grad(identity_loss, selected, retain_graph=True, allow_unused=True)
        b = torch.autograd.grad(5*raw, selected, retain_graph=True, allow_unused=True)
        output = []
        for name, x, y in zip(names, a, b):
            def describe(g):
                return dict(status='none' if g is None else ('zero' if not torch.count_nonzero(g) else 'nonzero'),
                            norm=None if g is None else float(g.detach().double().norm()))
            cosine = None
            if x is not None and y is not None and x.double().norm()>0 and y.double().norm()>0:
                cosine = float((x.double().flatten() @ y.double().flatten())/(x.double().norm()*y.double().norm()))
            output.append(dict(parameter=name, identity=describe(x), weighted_tgva=describe(y), cosine=cosine))
        self.write('gradient_epoch%d.json' % epoch, output)

    def save(self, epoch, train_seconds, epoch_batches=None):
        self.epoch_train_seconds += train_seconds
        budget = dict(epoch=epoch, epoch_actual_batches=epoch_batches, actual_batches=self.batches, optimizer_updates=self.updates,
            amp_skips=self.skips, training_loop_seconds=self.epoch_train_seconds,
            periodic_eval_seconds=self.eval_seconds, audit_seconds=self.audit_seconds,
            wall_seconds=self.previous_wall+time.time()-self.wall_start,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        meta = dict(self.manifest, epoch=epoch, budget=budget)
        payload = dict(state_dict=self.model.state_dict(), metadata=meta,
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(), rng=rng_state(), audit_panel=self.panel,
            initial_projection=self.initial_projection, initial_anchors=self.initial_anchors)
        temporary = os.path.join(self.c.OUTPUT_DIR, 'resume_latest.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, os.path.join(self.c.OUTPUT_DIR, 'resume_latest.pth'))
        if epoch == self.c.SOLVER.MAX_EPOCHS:
            torch.save(payload, os.path.join(self.c.OUTPUT_DIR, 'transformer_60.pth'))
            torch.save(dict(state_dict=deployment_state(self.model), metadata=meta),
                       os.path.join(self.c.OUTPUT_DIR, 'transformer_60_student.pth'))
        self.write('stage_budget.json', budget)
        with open(os.path.join(self.c.OUTPUT_DIR,'epoch_budget.jsonl'),'a',encoding='utf-8') as stream:
            stream.write(json.dumps(budget,ensure_ascii=False,allow_nan=False)+'\n')

    def restore(self, path):
        from yacs.config import CfgNode
        obj = torch.load(path, map_location='cpu', weights_only=False)
        meta = obj['metadata']
        for key in ('implementation','variant','original_clip_sha256','student','prompt'):
            if meta[key] != self.manifest[key]:
                raise ValueError('I1 resume manifest mismatch: '+key)
        # YAML canonicalizes tuple-valued legacy defaults to lists on reload.
        old, new = CfgNode.load_cfg(meta['config']), CfgNode.load_cfg(self.c.dump())
        for c in (old,new):
            c.OUTPUT_DIR = ''; c.ICLR_EXP.RESUME = ''
        if old != new:
            raise ValueError('I1 resume recipe differs')
        self.model.load_state_dict(obj['state_dict'], strict=True)
        self.optimizer.load_state_dict(obj['optimizer'])
        self.scheduler.load_state_dict(obj['scheduler'])
        self.scaler.load_state_dict(obj['scaler'])
        self.initial_projection, self.initial_anchors = obj['initial_projection'], obj['initial_anchors']
        self.panel = obj['audit_panel']
        b = meta['budget']
        self.batches, self.updates, self.skips = b['actual_batches'], b['optimizer_updates'], b['amp_skips']
        self.epoch_train_seconds, self.eval_seconds, self.audit_seconds = b['training_loop_seconds'], b['periodic_eval_seconds'], b['audit_seconds']
        self.previous_wall = b['wall_seconds']
        self.start_epoch = meta['epoch']+1
        restore_rng(obj['rng'])
