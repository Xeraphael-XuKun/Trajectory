"""D2: strict C0/M5 feature caches, original full-ranking metrics and query pairing."""
import argparse
import gc
import time
import torch.nn.functional as F
from common import *


def extract(paths, rows, role, out, args):
    folder = out/'features'/role
    folder.mkdir(parents=True, exist_ok=True)
    model, c = student(paths,args.device,role)
    assert c.TEST.FEAT_NORM == 'yes' and c.TEST.METRIC == 'sysu'
    assert not c.TEST.RE_RANKING and c.TEST.TOP_K_EVAL == 0
    if args.formal:
        from utils.runtime import configure_cudnn
        configure_cudnn(c.SOLVER.CUDNN_BENCHMARK)
        assert args.batch_size == c.TEST.IMS_PER_BATCH, 'Formal inference uses the historical TEST.IMS_PER_BATCH'
    snap = buffer_snapshot(model)
    identity = dict(model_sha256=digest(paths['m5_student' if role=='m5' else 'c0_student']),
        relative_paths=[r['relative_path'] for r in rows], environment=environment(),
        preprocessing=dict(size=list(c.INPUT.SIZE_TEST),mean=list(c.INPUT.PIXEL_MEAN),std=list(c.INPUT.PIXEL_STD)),
        dtype='float32', normalized=False, batch=args.batch_size,
        batch_order='original separate RGB/IR/Thermal loaders, query then gallery within each modality')
    p = folder/'manifest.json'
    progress = json.loads(p.read_text(encoding='utf-8')) if p.exists() else dict(signature=identity,completed=0)
    assert progress['signature']==identity, 'Feature cache provenance changed'
    offset = progress['completed']
    nq = sum(r['split']=='query' for r in rows)
    arrays = {}
    for readout in ('pre','post'):
        for split,n in [('query',nq),('gallery',len(rows)-nq)]:
            arrays[readout,split] = np.lib.format.open_memmap(folder/(readout+'_'+split+'.npy'),
                mode='r+' if offset else 'w+',dtype='float32',shape=(n,768))
    # Match the original three val_loaders' batch boundaries, while writing
    # caches by global query/gallery index. Eval BN never couples modalities.
    batches,consumed=[],0
    for modality in range(3):
        ids=[i for i,r in enumerate(rows) if r['modality']==modality]
        for start in range(0,len(ids),args.batch_size):
            batch=ids[start:start+args.batch_size]
            if consumed>=offset:
                batches.append(batch)
            else:
                assert consumed+len(batch)<=offset, 'Progress must end at an original batch boundary'
            consumed+=len(batch)
    loader = torch.utils.data.DataLoader(Images(rows,transforms(c)),batch_sampler=batches,
        num_workers=args.workers,pin_memory=True)
    start = time.monotonic()
    with torch.no_grad():
        for step,(images,ids) in enumerate(loader):
            images = images.to(args.device)
            pre = model.base(images)
            post = model.bottleneck(pre)
            assert torch.isfinite(pre).all() and torch.isfinite(post).all()
            if offset==0:
                model.neck_feat='after'
                torch.testing.assert_close(model(images,mode=1),post,atol=0,rtol=0)
                model.neck_feat='before'
                torch.testing.assert_close(model(images,mode=1),pre,atol=0,rtol=0)
            for readout,value in [('pre',pre),('post',post)]:
                a = value.cpu().numpy()
                for j,i in enumerate(ids.tolist()):
                    arrays[readout,'query' if i<nq else 'gallery'][i if i<nq else i-nq] = a[j]
            offset += len(ids)
            if (step+1)%128==0 or offset==len(rows):
                for a in arrays.values():
                    a.flush()
                progress['completed']=offset
                write_json(p,progress)
                print('D2_EXTRACT',role,offset,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    assert buffers_equal(model,snap)
    progress.update(buffers_unchanged=True,strict_load=True,full_visual_pass=not args.limit,
                    elapsed_this_invocation_seconds=time.monotonic()-start)
    write_json(p,progress)
    del model
    gc.collect()
    if args.device=='cuda':
        torch.cuda.empty_cache()


def protocols(query,gallery):
    masks = [('ALL',np.ones(len(query),dtype=bool),np.ones(len(gallery),dtype=bool))]
    qm,gm = [np.array([r['modality'] for r in rows]) for rows in (query,gallery)]
    qv,gv = [np.array([r['platform'] for r in rows]) for rows in (query,gallery)]
    for q in range(3):
        for g in range(3):
            masks.append((['RGB','IR','Thermal'][q]+'-'+['RGB','IR','Thermal'][g],qm==q,gm==g))
    for q in (1,0):
        for g in (1,0):
            masks.append((('A' if q else 'G')+'-'+('A' if g else 'G'),qv==q,gv==g))
    return [(name,np.flatnonzero(q),np.flatnonzero(g)) for name,q,g in masks]


def ranked_metrics(qf,gf,query,gallery,qi,gi,chunk):
    from utils.metrics import iter_euclidean_distance_chunks,eval_func,junk_mask
    qp,qc = [np.array([r[key] for r in query])[qi] for key in ('raw_pid','camid')]
    gp,gc = [np.array([r[key] for r in gallery])[gi] for key in ('raw_pid','camid')]
    results = []
    if not len(qi) or not len(gi):
        return results
    gf = gf[torch.tensor(gi)]
    for block,dist in enumerate(iter_euclidean_distance_chunks(qf[torch.tensor(qi)],gf,q_chunk_size=chunk)):
        start = block*chunk
        end = min(start+chunk,len(qi))
        # Same CPU torch.argsort path as the original metrics, no NumPy sorting substitution.
        indices = torch.argsort(dist,dim=1).numpy().astype(np.int32)
        local = []
        for j,order in enumerate(indices):
            k = start+j
            legal = order[~junk_mask(gp[order],gc[order],qp[k],qc[k],'sysu')]
            hits = (gp[legal]==qp[k]).astype(np.int32)
            positives = np.flatnonzero(hits)
            if not len(positives):
                results.append(dict(query_index=int(qi[k]),valid=False))
                continue
            cumulative = hits.cumsum()
            ap = float(((cumulative/np.arange(1,len(hits)+1))*hits).sum()/hits.sum())
            first,last = int(positives[0]+1),int(positives[-1]+1)
            d = dist[j].numpy()
            positive = legal[hits==1]
            negative = legal[hits==0]
            pg,ng = int(positive[0]),int(negative[0]) if len(negative) else None
            sorted_d = d[legal]
            row = dict(query_index=int(qi[k]),valid=True,ap=ap,inp=float(len(positives)/last),first_positive_rank=first,
                r1=int(first<=1),r5=int(first<=5),r10=int(first<=10),legal_positives=len(positives),
                top1_gallery_index=int(gi[legal[0]]),nearest_positive_gallery_index=int(gi[pg]),
                nearest_negative_gallery_index=int(gi[ng]) if ng is not None else None,
                positive_distance=float(d[pg]),negative_distance=float(d[ng]) if ng is not None else None,
                margin=float(d[ng]-d[pg]) if ng is not None else None,
                distance_tie_fraction=float(np.mean(sorted_d[1:]==sorted_d[:-1])) if len(sorted_d)>1 else 0.,
                top1_tied=bool(len(sorted_d)>1 and sorted_d[0]==sorted_d[1]))
            results.append(row)
            local.append(row)
        if local:
            # Only R1 is used in this cross-check. The historical evaluator
            # assumes enough legal candidates for max_rank=50, which a tiny
            # smoke gallery need not have. AP/INP still use the full ranking.
            cmc,map0,inp0 = eval_func(indices,qp[start:end],gp,qc[start:end],gc,max_rank=1,metric='sysu')
            assert abs(map0-np.mean([r['ap'] for r in local]))<1e-12
            assert abs(inp0-np.mean([r['inp'] for r in local]))<1e-12
            assert abs(float(cmc[0])-np.mean([r['r1'] for r in local]))<1e-6
        print('D2_RANK',end,'/',len(qi),'gallery',len(gi),flush=True)
    return results


def merge_pair(c0,m5,query,gallery):
    assert len(c0)==len(m5)
    paired = []
    for a,b in zip(c0,m5):
        assert a['query_index']==b['query_index'] and a['valid']==b['valid']
        i = a['query_index']
        row = dict(query_index=i,relative_path=query[i]['relative_path'],raw_pid=query[i]['raw_pid'],
                   camid=query[i]['camid'],modality=query[i]['modality'],platform=query[i]['platform'],valid=a['valid'])
        if a['valid']:
            row.update(delta_ap=b['ap']-a['ap'],rank_gain=a['first_positive_rank']-b['first_positive_rank'],
                conversion=('correct' if a['r1'] else 'wrong')+'->'+('correct' if b['r1'] else 'wrong'))
            for model,value in [('c0',a),('m5',b)]:
                row.update({model+'_'+k:v for k,v in value.items() if k not in ('query_index','valid')})
                for label in ('top1','nearest_positive','nearest_negative'):
                    idx=value[label+'_gallery_index']
                    g=gallery[idx] if idx is not None else None
                    row.update({model+'_'+label+'_'+k:g[k] if g else None
                        for k in ('relative_path','raw_pid','modality','platform','camid')})
            row.update(top1_changed=a['top1_gallery_index']!=b['top1_gallery_index'],
                nearest_positive_changed=a['nearest_positive_gallery_index']!=b['nearest_positive_gallery_index'],
                nearest_negative_changed=a['nearest_negative_gallery_index']!=b['nearest_negative_gallery_index'],
                delta_margin=b['margin']-a['margin'] if a['margin'] is not None and b['margin'] is not None else None)
        paired.append(row)
    # Invalid queries retain the same schema, with NA fields instead of zeros.
    keys = list(dict.fromkeys(k for row in paired for k in row))
    return [{k:r.get(k) for k in keys} for r in paired]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--resources',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--rank-chunk',type=int,default=64)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--extract-only',action='store_true')
    ap.add_argument('--formal',action='store_true')
    args=ap.parse_args()
    paths,out=resources(args.resources),Path(args.output)
    execution_mode(args.formal)
    assert not (args.formal and args.limit), 'Formal D2 requires the full test split'
    seed_diagnostics(formal=args.formal, config=m5_config(paths))
    query,gallery=index_split(paths,'query'),index_split(paths,'test')
    if args.formal:
        assert (len(query),len(gallery)) == (6405,93609)
    if args.limit:
        query,gallery=query[:args.limit],gallery[:args.limit]
    query=[dict(r,index=i,split='query') for i,r in enumerate(query)]
    gallery=[dict(r,index=i,split='gallery') for i,r in enumerate(gallery)]
    write_csv(out/'index_query.csv',query)
    write_csv(out/'index_gallery.csv',gallery)
    for role in ('c0','m5'):
        extract(paths,query+gallery,role,out,args)
    if args.extract_only:
        return
    groups = protocols(query,gallery)
    aggregates,paired_summary,errors,condition_rows,examples=[],[],[],[],[]
    for readout in ('post','pre'):
        features={role:[F.normalize(torch.tensor(np.load(out/'features'/role/(readout+'_'+split+'.npy'))),dim=1)
                        for split in ('query','gallery')] for role in ('c0','m5')}
        for name,qi,gi in groups:
            if name=='A-A':
                # Still evaluate validity; never silently fabricate a zero metric.
                pass
            values={}
            for role in ('c0','m5'):
                print('PROTOCOL',readout,name,role,flush=True)
                rows=ranked_metrics(*features[role],query,gallery,qi,gi,args.rank_chunk)
                values[role]=rows
                valid=[r for r in rows if r['valid']]
                metric=dict(model=role,readout=readout,protocol=name,queries=len(qi),valid_queries=len(valid),gallery=len(gi))
                metric.update({k:float(np.mean([r[key] for r in valid]))*100 if valid else None
                    for k,key in [('mAP','ap'),('Rank1','r1'),('Rank5','r5'),('Rank10','r10'),('mINP','inp')]})
                aggregates.append(metric)
            pair=merge_pair(values['c0'],values['m5'],query,gallery)
            write_csv(out/'per_query'/(readout+'_'+name+'.csv'),pair)
            valid=[r for r in pair if r['valid']]
            summary=dict(readout=readout,protocol=name,valid_queries=len(valid),
                improved=sum(r['delta_ap']>0 for r in valid),degraded=sum(r['delta_ap']<0 for r in valid),
                unchanged=sum(r['delta_ap']==0 for r in valid))
            for conv in ('wrong->correct','correct->wrong','correct->correct','wrong->wrong'):
                summary[conv]=sum(r['conversion']==conv for r in valid)
            for q in (0,.1,.25,.5,.75,.9,1):
                summary['delta_ap_pp_p'+str(int(q*100))]=float(np.quantile([r['delta_ap']*100 for r in valid],q)) if valid else None
            paired_summary.append(summary)
            if readout=='post' and name in ('ALL','G-A'):
                for r in valid:
                    if r['conversion'] in ('wrong->correct','correct->wrong'):
                        errors.append(dict(protocol=name,**r))
                conditions={}
                for r in valid:
                    key=(r['modality'],r['platform'],r['c0_top1_modality'],r['c0_top1_platform'],
                         r['m5_top1_modality'],r['m5_top1_platform'],r['conversion'])
                    conditions.setdefault(key,[]).append(r)
                for key,group in sorted(conditions.items()):
                    condition_rows.append(dict(protocol=name,query_modality=key[0],query_platform=key[1],
                        c0_top1_modality=key[2],c0_top1_platform=key[3],m5_top1_modality=key[4],m5_top1_platform=key[5],
                        conversion=key[6],queries=len(group),delta_ap_pp_mean=float(np.mean([r['delta_ap'] for r in group]))*100))
                for rule,selected in [('AP_gain',sorted([r for r in valid if r['delta_ap']>0],key=lambda r:(-r['delta_ap'],r['query_index']))[:5]),
                                      ('AP_loss',sorted([r for r in valid if r['delta_ap']<0],key=lambda r:(r['delta_ap'],r['query_index']))[:5])]:
                    examples.extend(dict(protocol=name,selection_rule=rule,**r) for r in selected)
            write_json(out/'metrics_reloaded.json',aggregates)
            write_csv(out/'paired_summary.csv',paired_summary)
        del features
        gc.collect()
    write_csv(out/'error_cases.csv',errors)
    write_csv(out/'top1_condition_summary.csv',condition_rows)
    write_csv(out/'query_examples.csv',examples)
    write_json(out/'manifest.json',dict(status='SMOKE_COMPLETED' if args.limit else ('COMPLETED' if args.formal else 'COMPLETED_WITH_CAVEAT'),
        formal_original_environment=args.formal,code_commit=code_commit(),batch_size=args.batch_size,
        full_test_pass=not args.limit,environment=environment(),query_count=len(query),gallery_count=len(gallery),
        metric='sysu',same_camera_all_excluded=True,sort='CPU torch.argsort full ranking',rank_chunk=args.rank_chunk,
        historical_scores_preserved=True,images_visually_inspected=False,raw_ap_range='[0,1]'))
    print('D2_COMPLETE',len(query),len(gallery),flush=True)


if __name__=='__main__':
    main()
