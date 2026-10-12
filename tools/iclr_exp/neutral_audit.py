"""Fixed clean train panel: C0 corrections and raw/neutral identity target quality."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.iclr_exp.audit import isolated_eval, summary
from model.m2_teacher import normalized_clip_logits
from loss.iclr_neutral import fixed_teacher_targets, kl_rows, target_quality


@torch.no_grad()
def run_audit(run,epoch):
    from PIL import Image
    from torchvision import transforms as T
    c,model,teacher,panel=run.c,run.model,run.teacher,run.panel
    root=Path(c.DATASETS.ROOT_DIR)/(c.DATASETS.SUBDIR or 'WHU-MARS')
    transform=T.Compose([T.Resize(c.INPUT.SIZE_TEST),T.ToTensor(),T.Normalize(c.INPUT.PIXEL_MEAN,c.INPUT.PIXEL_STD)])
    device=next(model.parameters()).device
    on_all,off_all,p_all,q_all,s_all,rho_all=[],[],[],[],[],{}
    with isolated_eval(model),isolated_eval(teacher):
        for start in range(0,len(panel['rows']),c.ICLR_EXP.AUDIT.BATCH_SIZE):
            rows=panel['rows'][start:start+c.ICLR_EXP.AUDIT.BATCH_SIZE]
            images=[]
            for row in rows:
                with Image.open(root/row['relative_path']) as image:images.append(transform(image.convert('RGB')))
            x=torch.stack(images).to(device)
            mods=torch.tensor([r['modality'] for r in rows],device=device)
            views=torch.tensor([r['platform'] for r in rows],device=device)
            pending,ratios,handles={},{},[]
            def capture(_,args,result):pending[int(args[0])]=result.detach()
            handles.append(model.base.token_trajectory.register_forward_hook(capture))
            def before(layer):
                def hook(_,args):
                    correction=pending[layer].double();previous=args[0].detach().double()-correction
                    ratios[layer]=torch.stack([
                        correction.flatten(1).norm(dim=1)/previous.flatten(1).norm(dim=1).clamp_min(1e-12),
                        correction[:,0].norm(dim=1)/previous[:,0].norm(dim=1).clamp_min(1e-12),
                        correction[:,1:].flatten(1).norm(dim=1)/previous[:,1:].flatten(1).norm(dim=1).clamp_min(1e-12)],dim=1).cpu()
                return hook
            for layer in range(1,12):handles.append(model.base.blocks[layer].register_forward_pre_hook(before(layer)))
            try:on=model.base(x)
            finally:
                for handle in handles:handle.remove()
            off=model.base(x,trajectory_gate=x.new_zeros(len(x)))
            gt=teacher.forward_pre_bn(x,mods,views)
            logp,logq=fixed_teacher_targets(normalized_clip_logits(gt,teacher.base.clip_proj,run.text),mods,views,run.shift)
            logs=normalized_clip_logits(on,model.base.clip_proj,run.text).div(2).log_softmax(-1)
            on_all.append(on.cpu());off_all.append(off.cpu());p_all.append(logp.cpu());q_all.append(logq.cpu());s_all.append(logs.cpu())
            for layer,value in ratios.items():rho_all.setdefault(layer,[]).append(value)
    on,off=torch.cat(on_all),torch.cat(off_all)
    logp,logq,logs=torch.cat(p_all),torch.cat(q_all),torch.cat(s_all)
    mods=torch.tensor([r['modality'] for r in panel['rows']]);views=torch.tensor([r['platform'] for r in panel['rows']]);targets=torch.tensor([r['pid'] for r in panel['rows']])
    projection=model.base.clip_proj.detach().float().cpu()
    gain=model.base.token_trajectory.gain.detach().double().cpu().flatten(1)
    output=dict(epoch=epoch,input='deterministic_clean_train; FP32 visual',rows=panel['rows'],
        target_quality=target_quality(logp,logq,targets,mods,views,run.paired.cpu()[targets,mods],logs),
        raw_teacher_kl_to_student=kl_rows(logp,logs).tolist(),neutral_kl_to_student=kl_rows(logq,logs).tolist(),
        on_off_cosine=F.cosine_similarity(on,off).tolist(),
        on_off_relative_distance=((on-off).norm(dim=1)/off.norm(dim=1).clamp_min(1e-12)).tolist(),
        joint_on_off_cosine=F.cosine_similarity(F.normalize(on@projection,dim=-1),F.normalize(off@projection,dim=-1)).tolist(),
        projection_relative_drift=float((projection-run.initial_projection).norm()/run.initial_projection.norm()),
        gain=dict(rms=gain.square().mean(-1).sqrt().tolist(),maxabs=gain.abs().amax(-1).tolist()),correction_ratios={})
    if run.c.ICLR_EXP.VARIANT == 'm5_paired':
        from loss.iclr_paired import branch_quality
        log_off=normalized_clip_logits(off,projection,run.text.cpu()).div(2).log_softmax(-1)
        output['off_target_quality']=target_quality(logp,logq,targets,mods,views,run.paired.cpu()[targets,mods],log_off)
        output['paired_branches']=branch_quality(logs,log_off,logp,logq,mods,views,run.shift.cpu())
    for layer,values in rho_all.items():
        values=torch.cat(values);entries=[]
        for m in range(3):
            for v in range(2):
                mask=(mods==m)&(views==v)
                if mask.any():entries.append(dict(modality=m,platform=v,count=int(mask.sum()),
                    all=summary(values[mask,0]),cls=summary(values[mask,1]),patch=summary(values[mask,2])))
        output['correction_ratios'][str(layer+1)]=entries
    directory=Path(c.OUTPUT_DIR)/'audit';directory.mkdir(parents=True,exist_ok=True)
    (directory/('epoch%d.json'%epoch)).write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return output
