"""Exactly one clean forward per training image; camera/platform/ID equal weights."""
import argparse
import datetime
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from tools.iclr_exp.neutral_assets import METHOD, json_digest


def estimate(camera_sums, counts, raw_ids, seed=20261012, folds=5):
    """Sums [Y,3,7,K], counts [Y,3,7]; missing camera means never enter averages."""
    camera = camera_sums/counts.clamp_min(1)[...,None]
    y, _, _, k = camera.shape
    platform = torch.zeros(y,3,2,k,dtype=torch.float64)
    present = torch.zeros(y,3,2,dtype=torch.bool)
    for v, cams in ((0,[0,1,2,3,4]), (1,[5,6])):
        valid = counts[:,:,cams]>0
        present[:,:,v] = valid.any(-1)
        platform[:,:,v] = (camera[:,:,cams]*valid[...,None]).sum(2)/valid.sum(2).clamp_min(1)[...,None]
    paired = present.all(-1)
    differences = platform[:,:,1]-platform[:,:,0]
    shifts, ids = [], []
    for m in range(3):
        selected = paired[:,m].nonzero().flatten()
        if not len(selected):
            raise ValueError('modality %d has no paired identities; cannot estimate shift'%m)
        ids.append(selected.tolist()); shifts.append(differences[selected,m].mean(0))
    shift = torch.stack(shifts)
    union = sorted(paired.any(-1).nonzero().flatten().tolist(), key=lambda i:raw_ids[i])
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(union),generator=generator).tolist()
    fold_ids = [[union[i] for i in permutation[f::folds]] for f in range(folds)]
    diagnostics = []
    for f, held_ids in enumerate(fold_ids):
        held = torch.zeros(y,dtype=torch.bool); held[held_ids] = True
        for m in range(3):
            fit, test = paired[:,m]&~held, paired[:,m]&held
            row = dict(fold=f, modality=m, fit_count=int(fit.sum()), heldout_count=int(test.sum()))
            if fit.any() and test.any():
                delta = differences[fit,m].mean(0)
                before = differences[test,m]; after = before-delta
                row.update(mean_gap_norm_before=float(before.mean(0).norm()),
                    mean_gap_norm_after=float(after.mean(0).norm()),
                    identity_gap_norm_before=before.norm(dim=-1).tolist(),
                    identity_gap_norm_after=after.norm(dim=-1).tolist(),
                    heldout_raw_ids=[raw_ids[i] for i in test.nonzero().flatten().tolist()])
                # Camera/view mean targets, not per-image probabilities.
                samples=platform[test,m].reshape(-1,k).float()
                corrected=(platform[test,m]+torch.tensor([.5,-.5],dtype=torch.float64)[None,:,None]*delta).reshape(-1,k).float()
                labels=test.nonzero().flatten().repeat_interleave(2)
                p=samples.div(2).log_softmax(-1); q=corrected.div(2).log_softmax(-1)
                idx=torch.arange(len(labels))
                row['platform_mean_target_diagnostic']=dict(raw_nll=float(-p[idx,labels].mean()),
                    neutral_nll=float(-q[idx,labels].mean()),raw_top1=float((p.argmax(-1)==labels).float().mean()),
                    neutral_top1=float((q.argmax(-1)==labels).float().mean()))
            else:
                row['status']='not_estimable_in_this_fold'
            diagnostics.append(row)
    return dict(group_counts=counts, camera_mean_logits=camera, platform_mean_logits=platform,
        platform_present=present, D_by_identity=differences, paired_mask=paired,
        paired_ids_by_modality=ids, num_paired_ids_by_modality=[len(x) for x in ids],
        shift_fp64=shift, shift_fp32=shift.float(),
        fit_residual_mean_norm=[float((differences[paired[:,m],m]-shift[m]).mean(0).norm()) for m in range(3)],
        crossfit_fold_ids=[[raw_ids[i] for i in group] for group in fold_ids], crossfit_summary=diagnostics)


class CleanTrain(torch.utils.data.Dataset):
    def __init__(self, rows, c):
        from torchvision import transforms as T
        self.rows=rows
        self.transform=T.Compose([T.Resize(c.INPUT.SIZE_TEST),T.ToTensor(),T.Normalize(c.INPUT.PIXEL_MEAN,c.INPUT.PIXEL_STD)])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        from PIL import Image
        path,y,cam,m=self.rows[i]
        with Image.open(path) as image: x=self.transform(image.convert('RGB'))
        return x,y,cam,m


def main():
    p=argparse.ArgumentParser(description='仅真实训练集一次确定性统计，I2/I3共享缓存')
    p.add_argument('--resources',required=True); p.add_argument('--config',default=str(ROOT/'configs/iclr_exp/m5_neutral.yml'))
    p.add_argument('--device',default='cuda'); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    from tools.iclr_exp.configuration import load_config
    from tools.iclr_exp.neutral_assets import load_resources, legacy_config, fingerprint
    from tools.iclr_exp.runtime import git
    from tools.m2.artifacts import train_source
    from model.m2_teacher import normalized_clip_logits
    c=load_config(variant=a.config,resources=a.resources)
    from utils.runtime import configure_cudnn,runtime_summary
    configure_cudnn(c.SOLVER.CUDNN_BENCHMARK)
    destination=Path(c.ICLR_EXP.ASSETS.NEUTRAL_STATS)
    if destination.exists(): raise ValueError('已有共享缓存，请复用；重新统计须显式指定新路径')
    teacher,text,source,meta=load_resources(c,a.device)
    rows,_=train_source(legacy_config(c))
    if len(rows)!=92133 or len(source['pid_map'])!=500:
        raise ValueError('真实训练集数量不同，请先复核；不能产出正式共享缓存')
    sums=torch.zeros(500,3,7,500,dtype=torch.float64); counts=torch.zeros(500,3,7,dtype=torch.long)
    loader=torch.utils.data.DataLoader(CleanTrain(rows,c),batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    seen=0
    with torch.no_grad():
        for images,y,cam,m in loader:
            images=images.to(a.device); mods=m.to(a.device); views=((cam==5)|(cam==6)).long().to(a.device)
            # No visual autocast: offline teacher FP32, accumulator FP64.
            g=teacher.forward_pre_bn(images,mods,views)
            logits=normalized_clip_logits(g,teacher.base.clip_proj,text,.07).double().cpu()
            logits-=logits.mean(-1,keepdim=True)
            flat=(y*3+m)*7+cam
            sums.view(-1,500).index_add_(0,flat,logits)
            counts.view(-1).index_add_(0,flat,torch.ones(len(y),dtype=torch.long))
            seen+=len(y)
            if seen%10000<len(y):print('clean train images: %d/%d'%(seen,len(rows)),flush=True)
    raw_ids=[p for p,y in sorted(source['pid_map'].items(),key=lambda r:r[1])]
    output=estimate(sums,counts,raw_ids)
    output.update(fingerprint(c,meta,source),schema_version=1,method_version=METHOD,complete=True,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        extraction_code_commit=git(['rev-parse','HEAD']).strip(), extraction_dirty_diff=git(['diff','HEAD']),
        merged_resource_config=c.dump(),runtime=runtime_summary(),num_train_images=seen,num_train_ids=500,
        inference_precision='FP32 visual; FP64 sums',base_temperature=.07,kd_temperature_reference=2.,
        logit_space='cosine/base_temperature; before distill temperature',class_centered=True,
        fit_input='deterministic_clean_train',estimator='camera_then_view_then_paired_identity_equal_mean',coefficient=1.)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');torch.save(output,temporary);temporary.replace(destination)
    brief={k:v for k,v in output.items() if not isinstance(v,torch.Tensor)}
    brief.update(shift_fp64=output['shift_fp64'].tolist(),shift_fp32=output['shift_fp32'].tolist(),
        group_counts=counts.tolist(),shift_norms=output['shift_fp64'].norm(dim=-1).tolist())
    destination.with_suffix('.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(destination, output['num_paired_ids_by_modality'],flush=True)


if __name__=='__main__':main()
