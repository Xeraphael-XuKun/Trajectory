"""Local fixed-weight pilot. Outputs are ignored; no formal trainer changes.

Gradient caching preserves the full PKM relationship pool and full-batch BN.
Chunked stochastic forwards are replayed exactly; they do not promise the same
random masks as a single monolithic CUDA forward.
"""
import argparse
import copy
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'work/check-deps'))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import cfg
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from loss.terminal_repair import terminal_repair
from model.terminal_repair import readonly_neck, replay, rng_state, set_rng
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

AMP_ENABLED = True

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def config(batch):
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/trajectory_C0.yml'))
    c.MODEL.PRETRAIN_PATH = 'E:/CFAN/weights/ViT-B-16.pt'
    c.DATASETS.ROOT_DIR = 'E:/CFAN/datasets'
    c.DATALOADER.NUM_WORKERS = 0; c.SOLVER.IMS_PER_BATCH = batch
    c.TEST.NECK_FEAT = 'after'
    return c


def checkpoints(base):
    base._pilot_checkpoint_enabled = True
    for block in base.blocks:
        original = block.forward
        def forward(x, chart=None, return_updates=False, f=original):
            if torch.is_grad_enabled() and base._pilot_checkpoint_enabled:
                return checkpoint(f, x, use_reentrant=False, preserve_rng_state=True)
            return f(x)
        block.forward = forward


def cache_features(model, images, chunk, need_reference):
    pieces, refs, states = [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        for x in images.split(chunk):
            state = rng_state(x.device); states.append(state)
            pieces.append(model.base(x))
            if need_reference:
                refs.append(replay(model.base, x, state, enabled=False))
    return torch.cat(pieces).detach().requires_grad_(), (torch.cat(refs) if refs else None), states


def replay_backward(model, images, chunk, states, gradients, gain_only=False):
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        for x, state, grad in zip(images.split(chunk), states, gradients.split(chunk)):
            z = replay(model.base, x, state, gain_only=gain_only)
            z.backward(grad)


def objective(model, features, ref, ids, cams, mods, loss_fn, weighting):
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        neck = model.bottleneck(features)
        main, ce, tri = loss_fn(model.classifier(neck), features, ids)
    if ref is None:
        return main, ce, tri, features.sum() * 0, {}
    repair, stats = terminal_repair(readonly_neck(ref, model.bottleneck),
                                   readonly_neck(features, model.bottleneck), ids, cams, mods,
                                   weighting=weighting)
    return main, ce, tri, repair, stats


def full_diagnostic(model, images, ids, cams, mods, loss_fn, chunk):
    """Actual full-backbone/gain gradients at lambda=1, before clipping/Adam."""
    bn = copy.deepcopy(model.bottleneck.state_dict()); start_rng = rng_state(images.device)
    f, ref, states = cache_features(model, images, chunk, True)
    main, ce, tri, aux, stats = objective(model, f, ref, ids, cams, mods, loss_fn, 'reference')
    scale = 128 if AMP_ENABLED else 1
    gm = torch.autograd.grad(main * scale, f, retain_graph=True)[0]
    ga = torch.autograd.grad(aux * scale, f)[0]
    row = {'ce': ce.item(), 'triplet': tri.item(), 'repair': aux.item(), **stats}
    del main, ce, tri, aux, ref, f
    model.zero_grad(set_to_none=True)
    replay_backward(model, images, chunk, states, ga)
    aux_grad = {n: p.grad.detach().float().cpu() / scale for n, p in model.base.named_parameters()
                if p.grad is not None}
    model.zero_grad(set_to_none=True)
    replay_backward(model, images, chunk, states, gm)
    for group in ('gain', 'backbone'):
        am = mm = dot = 0.
        for n, p in model.base.named_parameters():
            if (n.startswith('token_trajectory.') != (group == 'gain')) or n not in aux_grad:
                continue
            a = aux_grad[n]; b = p.grad.detach().float().cpu() / scale
            am += float(a.square().sum()); mm += float(b.square().sum()); dot += float((a*b).sum())
        row[group] = {'aux_norm': am**.5, 'main_norm': mm**.5,
                      'ratio': (am / mm)**.5 if mm else None,
                      'cosine': dot / (am * mm)**.5 if am * mm else None}
    model.zero_grad(set_to_none=True); model.bottleneck.load_state_dict(bn)
    set_rng(start_rng, images.device)
    return row


def sanity(model, batch, loss_fn, chunk):
    images, ids, cams, mods = batch
    # Full relationship pool is checked later; small paired forwards fit locally.
    images = images[:12]; ids = ids[:12]; cams = cams[:12]; mods = mods[:12]
    # Synthetic IDs/conditions guarantee valid relations for both on/off checks.
    ids = torch.arange(12, device='cuda') // 3
    cams = torch.tensor([0, 1, 5] * 4, device='cuda')
    mods = torch.tensor([0, 1, 2] * 4, device='cuda')
    bn = copy.deepcopy(model.bottleneck.state_dict()); seed(412)
    state = rng_state(images.device)
    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        outs = []; refs = []; states = []
        for x in images.split(chunk):
            st = rng_state(x.device); states.append(st); outs.append(model.base(x))
            with torch.no_grad(): refs.append(replay(model.base, x, st, enabled=False))
        f = torch.cat(outs); ref = torch.cat(refs)
        main, ce, tri, aux, _ = objective(model, f, ref, ids, cams, mods, loss_fn, 'reference')
        total = main + aux
    total.backward()
    expected = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    expected_f = f.detach().clone(); expected_loss = total.item()
    expected_bn = copy.deepcopy(model.bottleneck.state_dict()); expected_rng = rng_state(images.device)
    del f, ref, outs, refs, main, ce, tri, aux, total
    model.zero_grad(set_to_none=True); model.bottleneck.load_state_dict(bn); set_rng(state, images.device)
    f, ref, states = cache_features(model, images, chunk, True)
    main, ce, tri, aux, _ = objective(model, f, ref, ids, cams, mods, loss_fn, 'reference')
    total = main + aux; total.backward(); replay_backward(model, images, chunk, states, f.grad)
    dif = den = 0.
    for n, p in model.named_parameters():
        if n in expected:
            g = p.grad.detach().float().cpu(); dif += float((g - expected[n]).square().sum())
            den += float(expected[n].float().square().sum())
    result = {'cache_feature_max_error': float((f-expected_f).abs().max()),
              'cache_loss_error': abs(total.item()-expected_loss),
              'cache_gradient_relative_error': (dif/den)**.5,
              'cache_bn_equal': all(torch.equal(v, expected_bn[n]) for n,v in model.bottleneck.state_dict().items()),
              'cache_rng_equal': all(torch.equal(a,b) for a,b in zip(rng_state(images.device), expected_rng))}
    print(json.dumps({'cache_check':result}),flush=True)
    assert result['cache_feature_max_error'] == 0 and result['cache_loss_error'] == 0
    assert result['cache_gradient_relative_error'] < (0.005 if AMP_ENABLED else 1e-5)
    assert result['cache_bn_equal'] and result['cache_rng_equal']
    # Zero-gain on/off and no main RNG side effect.
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        st = rng_state(images.device); z = model.base(images[:chunk]); after = rng_state(images.device)
        z0 = replay(model.base, images[:chunk], st, enabled=False)
        result['zero_gain_on_off_error'] = float((z-z0).abs().max())
        result['reference_rng_equal'] = all(torch.equal(a,b) for a,b in zip(after,rng_state(images.device)))
    assert result['zero_gain_on_off_error'] == 0 and result['reference_rng_equal']
    # Gain-only full recurrence: all eleven rows receive gradients; backbone none.
    model.zero_grad(set_to_none=True)
    model.base._pilot_checkpoint_enabled = False
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED): z = replay(model.base, images[:chunk], st, gain_only=True)
    (z.float() * torch.linspace(-1,1,z.numel(),device=z.device).reshape_as(z)).sum().backward()
    result['gain_only_backbone_grads'] = [n for n,p in model.base.named_parameters()
                                        if not n.startswith('token_trajectory.') and p.grad is not None]
    result['gain_only_row_norms'] = model.base.token_trajectory.gain.grad.float().flatten(1).norm(dim=1).tolist()
    assert not result['gain_only_backbone_grads'] and min(result['gain_only_row_norms']) > 0
    model.base._pilot_checkpoint_enabled = True
    model.bottleneck.load_state_dict(bn); model.zero_grad(set_to_none=True)
    return result


def main():
    global AMP_ENABLED
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['sanity','warm','probe','train'], required=True)
    p.add_argument('--output', required=True)
    initial = p.add_mutually_exclusive_group()
    initial.add_argument('--start', help='Local model plus optimizer state')
    initial.add_argument('--weights', help='Read-only historical model for probe mode')
    p.add_argument('--steps', type=int, default=30); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--chunk', type=int, default=12); p.add_argument('--weight', type=float, default=1)
    p.add_argument('--data-seed', type=int, default=1234); p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--timeout-minutes', type=float, default=90)
    p.add_argument('--fp32', action='store_true')
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    AMP_ENABLED = not args.fp32
    if (out/'result.json').exists(): raise RuntimeError('Completed result exists; choose a new output directory')
    started = time.monotonic(); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True; torch.backends.cudnn.deterministic=True
    seed(1234); c=config(args.batch)
    loader,_,_,_,nc,cam,view=make_dataloader(c)
    model=make_model(c,nc,cam,view).cuda().train(); checkpoints(model.base)
    loss_fn,center=make_loss(c,nc); optimizer,_=make_optimizer(c,model,center)
    scheduler=create_scheduler(c,optimizer,iters_per_epoch=len(loader))
    scaler=torch.cuda.amp.GradScaler(init_scale=1024, enabled=AMP_ENABLED)
    step_offset=0
    if args.start:
        saved=torch.load(args.start,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']); step_offset=saved['updates']
        del saved; gc.collect()
    if args.weights:
        if args.mode != 'probe': raise ValueError('--weights is only a read-only probe')
        saved = torch.load(args.weights, map_location='cpu', weights_only=True)
        model.load_state_dict(saved, strict=True); del saved; gc.collect()
    seed(args.data_seed); iterator=iter(loader)
    report={'args':vars(args),'torch':torch.__version__,'python':sys.version,'gpu':torch.cuda.get_device_name(),
            'logical_batch':args.batch,'actual_images':args.batch*3,'loader_length':len(loader),
            'start_updates':step_offset,'records':[],'diagnostics':[]}
    def emit(row):
        print(json.dumps(row),flush=True)
        with (out/'trace.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
    emit({'event':'start','mode':args.mode,'weight':args.weight,'actual_images':args.batch*3,'start_updates':step_offset})
    for step in range(args.steps):
        if time.monotonic()-started > args.timeout_minutes*60: raise TimeoutError('Local pilot time limit reached')
        try: images,ids,cams=next(iterator)
        except StopIteration: iterator=iter(loader); images,ids,cams=next(iterator)
        images=torch.cat(images).cuda(); ids=ids.cuda().repeat(3); cams=torch.cat(cams).cuda()
        mods=torch.arange(3,device='cuda').repeat_interleave(args.batch)
        if args.mode=='sanity':
            report['sanity']=sanity(model,(images,ids,cams,mods),loss_fn,args.chunk); emit(report['sanity']); break
        if args.mode=='probe':
            row=full_diagnostic(model,images,ids,cams,mods,loss_fn,args.chunk)
            row['step']=step; report['diagnostics'].append(row); emit(row); continue
        if args.diagnostics and step in (0,args.steps-1):
            row=full_diagnostic(model,images,ids,cams,mods,loss_fn,args.chunk)
            row['step']=step; report['diagnostics'].append(row); emit({'diagnostic':row})
        optimizer.zero_grad(set_to_none=True)
        scheduler.step_update(step_offset+step)
        need_ref=args.mode=='train'
        f,ref,states=cache_features(model,images,args.chunk,need_ref)
        main_loss,ce,tri,aux,stats=objective(model,f,ref,ids,cams,mods,loss_fn,'reference')
        weight=args.weight if need_ref else 0
        total=main_loss+weight*aux; scaler.scale(total).backward()
        replay_backward(model,images,args.chunk,states,f.grad)
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        row={'step':step+1,'ce':float(ce),'triplet':float(tri),'repair':float(aux),
             'weighted_repair':weight*float(aux),'ratio_to_mean_main':weight*float(aux)/(float(main_loss)/2),
             'grad_norm_before_clip':float(norm),'amp_skipped':scaler.get_scale()<old_scale,
             'elapsed_seconds':time.monotonic()-started,'peak_gib':torch.cuda.max_memory_allocated()/1024**3,**stats}
        # Match the existing AMP trainer: GradScaler skips overflowed gradients.
        # Nonfinite loss or a nonfinite norm without a skipped update is invalid.
        assert torch.isfinite(total) and (torch.isfinite(norm) or row['amp_skipped'])
        report['records'].append(row); emit(row)
        del images,ids,cams,mods,f,ref,states,main_loss,ce,tri,aux,total
    if args.mode in ('warm','train'):
        torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},
                    'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'updates':step_offset+args.steps,'config':c.dump()},out/'local_state.pth')
    report['elapsed_seconds']=time.monotonic()-started
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit({'event':'completed','seconds':report['elapsed_seconds']})


if __name__=='__main__': main()
