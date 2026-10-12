"""Fresh strict reload; original SYSU rankings plus recoverable feature/index exports."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def summarize(rows, gallery_count):
    import numpy as np
    valid = [r for r in rows if r['valid']]
    result = dict(query_count=len(rows), valid_queries=len(valid), gallery_count=gallery_count)
    for key in ('AP','INP','R1','R5','R10'):
        result['mAP' if key=='AP' else ('mINP' if key=='INP' else key)] = float(np.mean([r[key] for r in valid])) if valid else None
    return result


def ranking_rows(qf,gf,queries,gallery,q_chunk=64):
    import numpy as np
    import torch
    from utils.metrics import iter_euclidean_distance_chunks, junk_mask, eval_func
    gp = np.array([r['raw_pid'] for r in gallery]); gc=np.array([r['camera_zero_based'] for r in gallery])
    qp = np.array([r['raw_pid'] for r in queries]); qc=np.array([r['camera_zero_based'] for r in queries])
    rows=[]; start=0
    for distances in iter_euclidean_distance_chunks(qf,gf,q_chunk):
        indices=torch.argsort(distances,dim=1).numpy()
        batch=[]
        for offset,order in enumerate(indices):
            i=start+offset
            keep=~junk_mask(gp[order],gc[order],qp[i],qc[i],'sysu')
            order=order[keep]; hits=np.flatnonzero(gp[order]==qp[i])+1
            row=dict(query_row=i,relative_path=queries[i]['relative_path'],raw_pid=int(qp[i]),
                modality=queries[i]['modality'],platform=queries[i]['platform'],camera_zero_based=int(qc[i]),
                valid=bool(len(hits)),positive_count=int(len(hits)),valid_gallery_count=len(order),
                AP=None,INP=None,first_positive_rank=None,R1=None,R5=None,R10=None,
                top1=gallery[int(order[0])] if len(order) else None)
            if len(hits):
                row.update(AP=float(np.mean(np.arange(1,len(hits)+1)/hits)),INP=float(len(hits)/hits[-1]),
                    first_positive_rank=int(hits[0]),R1=int(hits[0]<=1),R5=int(hits[0]<=5),R10=int(hits[0]<=10))
            rows.append(row); batch.append(row)
        # Compare every chunk with the unchanged project evaluator.
        if any(r['valid'] for r in batch):
            cmc, mAP, mINP=eval_func(indices,qp[start:start+len(indices)],gp,qc[start:start+len(indices)],gc,max_rank=1)
            calculated=summarize(batch,len(gallery))
            if not (abs(mAP-calculated['mAP'])<1e-10 and abs(mINP-calculated['mINP'])<1e-10 and abs(float(cmc[0])-calculated['R1'])<1e-6):
                raise AssertionError('exported per-query metrics differ from original evaluator')
        start+=len(indices)
    return rows


def all_groups(qf,gf,qindex,gindex):
    import torch
    rows=ranking_rows(qf,gf,qindex,gindex)
    result=dict(ALL=summarize(rows,len(gindex)),query_conditions_full_gallery={},modality_pairs={},platform_pairs={},condition_pairs={})
    def subset(name, selector_q, selector_g):
        qi=[i for i,r in enumerate(qindex) if selector_q(r)]
        gi=[i for i,r in enumerate(gindex) if selector_g(r)]
        if not qi or not gi:
            return summarize([],len(gi))
        sub=ranking_rows(qf[torch.tensor(qi)],gf[torch.tensor(gi)],[qindex[i] for i in qi],[gindex[i] for i in gi])
        return summarize(sub,len(gi))
    for m in range(3):
        for n in range(3):
            key='%d_to_%d'%(m,n)
            result['modality_pairs'][key]=subset(key,lambda r:r['modality']==m,lambda r:r['modality']==n)
    for v in range(2):
        for w in range(2):
            key=('G','A')[v]+'_to_'+('G','A')[w]
            result['platform_pairs'][key]=subset(key,lambda r:r['platform']==v,lambda r:r['platform']==w)
    for m in range(3):
        for v in range(2):
            key='%d_%d'%(m,v)
            result['query_conditions_full_gallery'][key]=summarize([r for r in rows if r['modality']==m and r['platform']==v],len(gindex))
            for n in range(3):
                for w in range(2):
                    key='%d_%d_to_%d_%d'%(m,v,n,w)
                    result['condition_pairs'][key]=subset(key,lambda r:r['modality']==m and r['platform']==v,lambda r:r['modality']==n and r['platform']==w)
    return result,rows


def main():
    p=argparse.ArgumentParser(description='I1 epoch60独立重载与完整排序导出')
    p.add_argument('--resources',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--full-checkpoint'); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int,default=0,help='每模态每split的小样本数量；非正式评测')
    p.add_argument('--allow-nonfinal',action='store_true',help='仅用于本地预检工件')
    a=p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms as T
    from torch.utils.data import DataLoader
    from datasets.whu_mars import WHU_MARS
    from datasets.bases import ImageDatasetTest
    from datasets.make_dataloader import val_collate_fn
    from model import make_model
    from tools.iclr_exp.configuration import load_config
    from tools.iclr_exp.runtime import VERSION,git
    from tools.m2.artifacts import file_digest
    c=load_config(resources=a.resources)
    obj=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    meta=obj['metadata']
    if meta['implementation']!=VERSION or meta['variant']!='tgva_ref' or meta['student']!=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=[11,1,129,768]):
        raise ValueError('I1 checkpoint/method mismatch')
    if meta['epoch']!=60 and not a.allow_nonfinal:
        raise ValueError('formal evaluation must reload epoch60')
    if meta['original_clip_sha256']!=file_digest(c.MODEL.PRETRAIN_PATH):
        raise ValueError('original CLIP source differs')
    c=c.clone(); c.defrost(); c.ICLR_EXP.ENABLED=False; c.freeze()
    model=make_model(c,500,7,2)
    state={k:v for k,v in obj['state_dict'].items() if not k.startswith('iclr_tgva.')}
    model.load_state_dict(state,strict=True)
    if a.full_checkpoint:
        full=torch.load(a.full_checkpoint,map_location='cpu',weights_only=False)
        full_state={k:v for k,v in full['state_dict'].items() if not k.startswith('iclr_tgva.')}
        if full['metadata']!=meta or set(full_state)!=set(state) or not all(torch.equal(v,full_state[k]) for k,v in state.items()):
            raise ValueError('deployment and complete checkpoint tensors/metadata differ')
        del full,full_state
    del obj,state
    model.to(a.device).eval()
    ds=WHU_MARS(root=c.DATASETS.ROOT_DIR,modalities=list(c.DATASETS.MODALITIES),protocol='ALL',subdir=c.DATASETS.SUBDIR)
    if not a.limit and (ds.num_query_imgs!=6405 or ds.num_gallery_imgs!=93609):
        raise ValueError('WHU-MARS-1000 evaluation counts differ; investigate data before formal evaluation')
    transform=T.Compose([T.Resize(c.INPUT.SIZE_TEST),T.ToTensor(),T.Normalize(c.INPUT.PIXEL_MEAN,c.INPUT.PIXEL_STD)])
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    data_root=Path(ds.dataset_dir)
    features={'query':{'pre':[],'post':[]},'gallery':{'pre':[],'post':[]}}
    index={'query':[],'gallery':[]}
    readout_checked=False
    for m,name in enumerate(c.DATASETS.MODALITIES):
        for split,source in (('query',ds.query),('gallery',ds.gallery)):
            records=source[name][:a.limit] if a.limit else source[name]
            loader=DataLoader(ImageDatasetTest({name:records},name,transform),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,collate_fn=val_collate_fn)
            for images,*_ in loader:
                with torch.no_grad():
                    x=images.to(a.device); g=model.base(x); f=model.bottleneck(g)
                    if not readout_checked:
                        model.neck_feat='before'; before=model(x,mode=m+1)
                        model.neck_feat='after'; after=model(x,mode=m+1)
                        if not (torch.equal(g,before) and torch.equal(f,after)):
                            raise AssertionError('one-pass pre/post extraction differs from original readouts')
                        readout_checked=True
                features[split]['pre'].append(g.float().cpu()); features[split]['post'].append(f.float().cpu())
            for path,pid,cam,_ in records:
                index[split].append(dict(relative_path=Path(path).relative_to(data_root).as_posix(),raw_pid=pid,
                    camera_zero_based=cam,modality=m,platform=int(cam in (5,6)),split=split,row_index=len(index[split])))
    for split in ('query','gallery'):
        with (output/(split+'_index.jsonl')).open('w',encoding='utf-8') as f:
            for row in index[split]:f.write(json.dumps(row)+'\n')
        for neck in ('post','pre'):
            features[split][neck]=F.normalize(torch.cat(features[split][neck]),dim=1)
            np.save(output/(split+'_features_'+neck+'.npy'),features[split][neck].numpy())
    result=dict(implementation=VERSION,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=file_digest(a.checkpoint),
        evaluation_commit=git(['rev-parse','HEAD']).strip(),config=c.dump(),epoch=meta['epoch'],
        formal=not bool(a.limit or a.allow_nonfinal),metric='sysu',precision='FP32 no AMP',
        query_index_sha256=file_digest(output/'query_index.jsonl'),gallery_index_sha256=file_digest(output/'gallery_index.jsonl'),readout_equivalence=True)
    for neck in ('post','pre'):
        scores,rows=all_groups(features['query'][neck],features['gallery'][neck],index['query'],index['gallery'])
        result[neck]=scores
        with (output/('per_query_'+neck+'.jsonl')).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        # Preserve the restricted G->A task separately from ground queries to ALL gallery.
        qi=[i for i,r in enumerate(index['query']) if r['platform']==0]
        gi=[i for i,r in enumerate(index['gallery']) if r['platform']==1]
        ga_rows=[]
        if qi and gi:
            ga_rows=ranking_rows(features['query'][neck][torch.tensor(qi)],features['gallery'][neck][torch.tensor(gi)],
                [index['query'][i] for i in qi],[index['gallery'][i] for i in gi])
        with (output/('per_query_G_to_A_'+neck+'.jsonl')).open('w',encoding='utf-8') as f:
            for row in ga_rows:f.write(json.dumps(row)+'\n')
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(formal=result['formal'],post=result['post']['ALL'],pre=result['pre']['ALL']),ensure_ascii=False))


if __name__=='__main__':main()
