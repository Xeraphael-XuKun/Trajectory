"""D1: actual pre-BN/projected KD on train IDs, deterministic and augmented eval."""
import argparse
import time
from collections import defaultdict

import torch.nn.functional as F
from common import *


def infer(rows, c, tmodel, smodel, out, batch, workers, augmented=False):
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(rows=len(rows), teacher=digest(ARGS_PATHS['m5_teacher']),
                     student=digest(ARGS_PATHS['m5_student']), text=digest(ARGS_PATHS['m5_text']),
                     environment=environment(), augmented=augmented, dtype='float32', batch=batch)
    signature['image_order_and_seeds'] = [(r['relative_path'], r.get('augmentation_seed')) for r in rows]
    # JSON represents tuples as lists; use the same representation on resume.
    signature['image_order_and_seeds'] = [list(v) for v in signature['image_order_and_seeds']]
    progress_path = out/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(signature=signature, completed=0)
    assert progress['signature'] == signature, 'Cache provenance changed'
    offset = progress['completed']
    arrays = [np.lib.format.open_memmap(out/('logits_'+name+'_float32.npy'),
              mode='r+' if offset else 'w+', dtype='float32', shape=(len(rows), 500)) for name in ('teacher', 'student_m5r1')]
    transform = transforms(c, augmented)
    ds = AugmentedImages(rows, transform) if augmented else Images(rows, transform)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(offset, len(rows))),
        batch_size=batch, num_workers=workers, shuffle=False, pin_memory=True)
    text = load(ARGS_PATHS['m5_text'])['text_bank'].float().to(ARGS.device)
    from model.m2_teacher import normalized_clip_logits
    started = time.monotonic()
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(ARGS.device)
            mod = torch.tensor([rows[int(i)]['modality'] for i in ids], device=ARGS.device)
            view = torch.tensor([rows[int(i)]['platform'] for i in ids], device=ARGS.device)
            gt = tmodel.forward_pre_bn(images, mod, view)
            lt = normalized_clip_logits(gt, tmodel.base.clip_proj, text, c.M2.TEXT_TEMPERATURE)
            del gt
            gs = smodel.base(images)
            ls = normalized_clip_logits(gs, smodel.base.clip_proj, text, c.M2.TEXT_TEMPERATURE)
            assert torch.isfinite(lt).all() and torch.isfinite(ls).all()
            arrays[0][ids.numpy()] = lt.cpu().numpy()
            arrays[1][ids.numpy()] = ls.cpu().numpy()
            offset += len(ids)
            if offset % (batch*64) == 0 or offset == len(rows):
                for a in arrays:
                    a.flush()
                progress['completed'] = offset
                write_json(progress_path, progress)
                print('D1', out.name, offset, '/', len(rows), 'seconds', round(time.monotonic()-started, 1), flush=True)
    return arrays


class AugmentedImages(Images):
    def __getitem__(self, i):
        seed = self.rows[i]['augmentation_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return super().__getitem__(i)


def metrics(rows, t, s, td):
    output = []
    for start in range(0, len(rows), 1024):
        end = min(start+1024, len(rows))
        ut, us = torch.tensor(np.asarray(t[start:end]))/td, torch.tensor(np.asarray(s[start:end]))/td
        y = torch.tensor([r['train_pid'] for r in rows[start:end]])
        ix = torch.arange(len(y))
        lt, ls = F.log_softmax(ut, -1), F.log_softmax(us, -1)
        keep = torch.ones_like(ut, dtype=torch.bool)
        keep[ix, y] = False
        tn, sn = ut[keep].reshape(-1,499), us[keep].reshape(-1,499)
        bt = F.log_softmax(torch.stack([ut[ix,y], tn.logsumexp(-1)],-1),-1)
        bs = F.log_softmax(torch.stack([us[ix,y], sn.logsumexp(-1)],-1),-1)
        nt, ns = F.log_softmax(tn,-1), F.log_softmax(sn,-1)
        binary = (bt.exp()*(bt-bs)).sum(-1)
        tail = (nt.exp()*(nt-ns)).sum(-1)
        mass = bt[:,1].exp()
        full = (lt.exp()*(lt-ls)).sum(-1)
        torch.testing.assert_close(binary+mass*tail, full, rtol=1e-5, atol=1e-6)
        at, ass = bt[:,0].exp(), bs[:,0].exp()
        pt, ps = ut.argmax(-1), us.argmax(-1)
        fields = dict(teacher_correct=pt==y, student_correct=ps==y,
            teacher_true_rank=1+(ut>ut[ix,y,None]).sum(-1), student_true_rank=1+(us>us[ix,y,None]).sum(-1),
            teacher_nll=-lt[ix,y], student_nll=-ls[ix,y], teacher_p_y=at, student_p_y=ass,
            delta_p_y=ass-at, active=bs[:,0]<bt[:,0],
            teacher_entropy=-(lt.exp()*lt).sum(-1), student_entropy=-(ls.exp()*ls).sum(-1),
            teacher_tail_entropy=-(nt.exp()*nt).sum(-1), student_tail_entropy=-(ns.exp()*ns).sum(-1),
            teacher_tail_uniform_kl=np.log(499)+(nt.exp()*nt).sum(-1),
            student_tail_uniform_kl=np.log(499)+(ns.exp()*ns).sum(-1),
            binary_raw=binary, tail_raw=tail, teacher_tail_mass=mass,
            weighted_tail_raw=mass*tail, full_kd_raw=full, kd_with_Td2=full*td*td,
            flat_tail_argmax_rewritten=(pt!=y)&(at>1/500))
        values = {k:v.numpy() for k,v in fields.items()}
        for i, r in enumerate(rows[start:end]):
            result = dict(index=r['index'], relative_path=r['relative_path'], raw_pid=r['raw_pid'], train_pid=r['train_pid'],
                          modality=r['modality'], camid=r['camid'], platform=r['platform'])
            result.update({k: float(v[i]) for k,v in values.items()})
            output.append(result)
    return output


def summaries(rows):
    grouped = []
    masks = [('ALL', lambda r:True)]
    masks += [(m, lambda r,k=k:r['modality']==k) for k,m in enumerate(('RGB','IR','Thermal'))]
    masks += [(v, lambda r,k=k:r['platform']==k) for k,v in enumerate(('G','A'))]
    masks += [(m+'_'+v, lambda r,k=k,j=j:r['modality']==k and r['platform']==j)
              for k,m in enumerate(('RGB','IR','Thermal')) for j,v in enumerate(('G','A'))]
    fields = [k for k in rows[0] if k not in ('index','relative_path','raw_pid','train_pid','modality','camid','platform')]
    for name, mask in masks:
        selected = [r for r in rows if mask(r)]
        if not selected:
            continue
        bypid = defaultdict(list)
        for r in selected:
            bypid[r['train_pid']].append(r)
        for key in fields:
            vals = np.array([r[key] for r in selected])
            macro = np.mean([np.mean([r[key] for r in v]) for v in bypid.values()])
            grouped.append(dict(group=name, metric=key, images=len(vals), identities=len(bypid),
                camera_count=len({r['camid'] for r in selected}), micro_mean=float(vals.mean()),
                identity_macro_mean=float(macro), median=float(np.median(vals)),
                p10=float(np.quantile(vals,.1)), p90=float(np.quantile(vals,.9))))
    return grouped


def augmentation_rows(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r['train_pid'],r['modality'],r['platform'])].append(r)
    chosen = []
    rng = np.random.default_rng(1234)
    for cell in sorted(cells):
        candidates = cells[cell]
        first = candidates[0]
        second = next((r for r in candidates[1:] if r['camid']!=first['camid']), None)
        if second is None and len(candidates)>1:
            second = candidates[1]
        for r in [first]+([second] if second is not None else []):
            chosen.append(dict(r, seed0=int(rng.integers(0,2**31)), seed1=int(rng.integers(0,2**31))))
    assert len(chosen)<=6000
    expanded = []
    for r in chosen:
        for j in (0,1):
            expanded.append(dict(r,index=len(expanded),augmentation_seed=r['seed'+str(j)],augmentation=j))
    return chosen, expanded


def main():
    global ARGS, ARGS_PATHS
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='Nonzero is a smoke check, never a full D1 result')
    ap.add_argument('--formal', action='store_true')
    ARGS = ap.parse_args()
    ARGS_PATHS, out = resources(ARGS.resources), Path(ARGS.output)
    execution_mode(ARGS.formal)
    assert not (ARGS.formal and ARGS.limit), 'Formal D1 requires the full train split'
    seed_diagnostics(formal=ARGS.formal, config=m5_config(ARGS_PATHS))
    start = time.monotonic()
    rows = index_split(ARGS_PATHS, 'train')
    if ARGS.formal:
        assert len(rows)==92133 and len({r['train_pid'] for r in rows})==500
    if ARGS.limit:
        # Cover each modality rather than selecting just the first RGB rows.
        selected = []
        for m in range(3):
            selected.extend([r for r in rows if r['modality']==m][:max(1,ARGS.limit//3)])
        rows = selected[:ARGS.limit]
        rows = [dict(r,index=i) for i,r in enumerate(rows)]
    tmodel, tmeta = teacher(ARGS_PATHS, ARGS.device)
    smodel, c = student(ARGS_PATHS, ARGS.device)
    assert not tmodel.training and not smodel.training
    tsnap, ssnap = buffer_snapshot(tmodel), buffer_snapshot(smodel)
    arrays = infer(rows, c, tmodel, smodel, out, ARGS.batch_size, ARGS.workers)
    write_csv(out/'train_index.csv', rows)
    report = metrics(rows,*arrays,c.M2.KD_TEMPERATURE)
    write_csv(out/'per_image_metrics.csv',report)
    group = summaries(report)
    write_csv(out/'group_summary.csv',group)
    write_csv(out/'per_image_metrics_Td1_supplement.csv',metrics(rows,*arrays,1.))
    four = []
    for tc in (0,1):
        for sc in (0,1):
            selected = [r for r in report if r['teacher_correct']==tc and r['student_correct']==sc]
            four.append(dict(teacher_correct=tc,student_correct=sc,images=len(selected),
                active_fraction=np.mean([r['active'] for r in selected]).item() if selected else None))
    write_csv(out/'correctness_active_quadrants.csv',four)
    subset, augrows = augmentation_rows(rows)
    write_csv(out/'augmentation_subset.csv',subset)
    augarrays = infer(augrows,c,tmodel,smodel,out/'augmentation',ARGS.batch_size,ARGS.workers,True)
    augmetrics = metrics(augrows,*augarrays,c.M2.KD_TEMPERATURE)
    write_csv(out/'augmentation/per_image_metrics.csv', augmetrics)
    write_csv(out/'augmentation/group_summary.csv', summaries(augmetrics))
    augresults = []
    for i,r in enumerate(subset):
        i0,i1 = 2*i,2*i+1
        lt0,lt1 = [F.log_softmax(torch.tensor(np.asarray(augarrays[0][j]))/2,-1) for j in (i0,i1)]
        lm = torch.logaddexp(lt0,lt1)-np.log(2)
        js = .5*((lt0.exp()*(lt0-lm)).sum()+(lt1.exp()*(lt1-lm)).sum()).item()
        augresults.append(dict(index=r['index'], relative_path=r['relative_path'], train_pid=r['train_pid'],
            modality=r['modality'],platform=r['platform'],teacher_delta_p_y=augmetrics[i1]['teacher_p_y']-augmetrics[i0]['teacher_p_y'],
            student_delta_p_y=augmetrics[i1]['student_p_y']-augmetrics[i0]['student_p_y'],
            teacher_argmax_flip=int(lt0.argmax()!=lt1.argmax()),teacher_js=js,
            active_flip=int(augmetrics[i0]['active']!=augmetrics[i1]['active'])))
    write_csv(out/'augmentation_metrics.csv',augresults)
    assert buffers_equal(tmodel,tsnap) and buffers_equal(smodel,ssnap)
    coverage = [dict(train_pid=y,conditions=len({(r['modality'],r['platform']) for r in rows if r['train_pid']==y}))
                for y in sorted({r['train_pid'] for r in rows})]
    write_csv(out/'identity_condition_coverage.csv',coverage)
    manifest = dict(status='SMOKE_COMPLETED' if ARGS.limit else ('COMPLETED' if ARGS.formal else 'COMPLETED_WITH_CAVEAT'),
        formal_original_environment=ARGS.formal, code_commit=code_commit(),
        full_train_pass=not ARGS.limit, images=len(rows), augmentation_images=len(subset),
        augmentation_passes=2, student_mode='eval',teacher_mode='eval', buffers_unchanged=True,
        base_temperature=.07,distillation_temperature=2., environment=environment(),
        teacher_compatibility=getattr(tmodel,'diagnostic_compatibility',{}),
        elapsed_seconds=time.monotonic()-start, gpu_peak_bytes=torch.cuda.max_memory_allocated() if ARGS.device=='cuda' else None,
        overall={r['metric']:r for r in group if r['group']=='ALL'})
    write_json(out/'manifest.json',manifest)
    print('D1_COMPLETE',manifest['status'],len(rows),round(manifest['elapsed_seconds'],2),flush=True)


if __name__=='__main__':
    main()
