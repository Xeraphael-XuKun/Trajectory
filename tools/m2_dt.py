"""Separate DT checks/evaluation; full training uses the existing train.py."""
import argparse
import copy
import logging
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from config import cfg as defaults
from utils.direct_text import validate_config,load_text,text_ce,write_json,load_deployment,save_training,restore

def configure(args):
    c=defaults.clone();c.merge_from_file(args.config);c.merge_from_list(args.opts)
    validate_config(c)
    return c

def dataset(c):
    from datasets.whu_mars import WHU_MARS
    return WHU_MARS(root=c.DATASETS.ROOT_DIR,modalities=c.DATASETS.MODALITIES,
        protocol=c.DATASETS.PROTOCOL,subdir=c.DATASETS.SUBDIR,verbose=False)

def preflight(args,c):
    from model import make_model
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from train import set_seed
    from utils.direct_text import state_digest,config_diff
    from tools.m2.artifacts import code_commit
    torch.set_num_threads(2)
    data=dataset(c)
    text,resources=load_text(c,data.train,args.device,data.num_train_pids)
    assert data.num_train_pids==500 and sum(len(v) for v in data.train.values())==92133
    n=args.images_per_modality
    if args.formal:
        assert args.device=='cuda' and n==64,'Server formal preflight must cover the full 192-image batch'
    # Same seed and no auxiliary creation: DT must not change the visual state or RNG.
    off=c.clone();off.M2_DIRECT_TEXT.ENABLED=False
    off.M2_DIRECT_TEXT.TEXT_FEATURES_PATH='/intentionally/nonexistent/disabled-path.pt'
    set_seed(c.SOLVER.SEED,c.SOLVER.CUDNN_BENCHMARK)
    plain=make_model(off,500,camera_num=7,view_num=3)
    plain_state={k:v.detach().cpu().clone() for k,v in plain.state_dict().items()}
    plain_rng=torch.random.get_rng_state().clone()
    del plain
    set_seed(c.SOLVER.SEED,c.SOLVER.CUDNN_BENCHMARK)
    model=make_model(c,500,camera_num=7,view_num=3).to(args.device)
    assert torch.equal(plain_rng,torch.random.get_rng_state())
    assert all(torch.equal(v.cpu(),plain_state[k]) for k,v in model.state_dict().items())
    initial_hash=state_digest(model.state_dict())
    assert not model.base.clip_proj.requires_grad
    # Direct P0 correspondence, independent of the filename/hash check.
    original=torch.jit.load(c.MODEL.PRETRAIN_PATH,map_location='cpu').state_dict()['visual.proj']
    assert torch.equal(original.float(),model.base.clip_proj.detach().cpu().float())
    del original
    assert not any('teacher' in name or 'lora' in name or 'text_encoder' in name for name,_ in model.named_modules())
    x=[torch.randn(n,3,256,128,device=args.device) for _ in range(3)]
    labels=(torch.arange(n,device=args.device)//4) if n>=4 else torch.arange(n,device=args.device)
    target=labels.repeat(3)
    cams=[torch.zeros(n,dtype=torch.long,device=args.device) for _ in range(3)]
    model.train();model.zero_grad(set_to_none=True)
    rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state() if args.device=='cuda' else None
    bn0=model.bottleneck.num_batches_tracked.item()
    with torch.cuda.amp.autocast(enabled=args.device=='cuda'):
        score,g,neck=model(x,labels,cams)
    assert g.shape==(3*n,768) and model.bottleneck.num_batches_tracked.item()==bn0+1
    g.retain_grad()
    aux,logits=text_ce(g,model.base.clip_proj,text,target)
    assert logits.shape==(3*n,500) and logits.dtype==torch.float32
    gain=model.base.token_trajectory.gain
    aux_grads=torch.autograd.grad(aux,[g,gain,model.bottleneck.weight,model.classifier.weight],
        allow_unused=True,retain_graph=True)
    assert torch.isfinite(aux_grads[0]).all() and aux_grads[0].abs().sum()>0
    assert aux_grads[1] is not None and torch.isfinite(aux_grads[1]).all() and aux_grads[1].abs().sum()>0
    assert aux_grads[2] is None and aux_grads[3] is None
    assert model.base.clip_proj.grad is None and text.grad is None
    # Loss computation itself must not consume RNG or add another visual/BN forward.
    after_rng=torch.random.get_rng_state().clone()
    after_bn=model.bottleneck.state_dict()
    model.load_state_dict(plain_state,strict=True)
    torch.random.set_rng_state(rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
    with torch.cuda.amp.autocast(enabled=args.device=='cuda'):
        score2,g2,neck2=model(x,labels,cams)
    for a,b in [(score,score2),(g,g2),(neck,neck2)]:torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert torch.equal(after_rng,torch.random.get_rng_state())
    # Backward on the fresh graph, matching the original base loss.
    del score,g,neck,aux,logits,aux_grads
    aux,logits=text_ce(g2,model.base.clip_proj,text,target)
    if args.device=='cuda':
        from loss import make_loss
        base_loss,center=make_loss(c,500)
        with torch.cuda.amp.autocast():
            base,ce,tri=base_loss(score2,g2,target)
    else:
        from loss.triplet_loss import TripletLoss
        from loss.center_loss import CenterLoss
        center=CenterLoss(500,2048,use_gpu=False)
        ce=F.cross_entropy(score2,target);tri=TripletLoss()(g2,target)[0];base=ce+tri
    optimizer,_=make_optimizer(c,model,center)
    scheduler=create_scheduler(c,optimizer,iters_per_epoch=469)
    scaler=torch.cuda.amp.GradScaler(enabled=args.device=='cuda')
    total=base+.5*aux
    scaler.scale(total).backward();scaler.unscale_(optimizer)
    assert model.classifier.weight.grad is not None and model.bottleneck.weight.grad is not None
    assert model.base.clip_proj.grad is None and torch.isfinite(gain.grad).all()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    assert scaler.get_scale()>=scale and torch.isfinite(norm)
    assert torch.equal(model.base.clip_proj.detach().cpu(),plain_state['base.clip_proj'])
    model.dt_metadata=dict(implementation='m2-5-dt-v1',variant='direct_identity_text_c0',config=c.dump(),
        resources=resources,student=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=list(gain.shape)))
    model.eval()
    with torch.no_grad():
        model.neck_feat='before';pre=model(x[0],mode=1).detach().clone()
        model.neck_feat='after';post=model(x[0],mode=1).detach().clone()
    # Full restore and strict deployment reload, without depending on a text file.
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output)) as td:
        save_c=c.clone();save_c.OUTPUT_DIR=td
        saved=save_training(save_c,60,model,optimizer,scheduler,scaler,1,1,0,0.,None)
        deploy=Path(td)/'transformer_60_student.pth'
        obj=load_deployment(model,deploy)
        assert all(torch.equal(v.cpu(),obj['state_dict'][k].cpu()) for k,v in saved['state_dict'].items())
        resume=c.clone();resume.OUTPUT_DIR=td;resume.M2_DIRECT_TEXT.RESUME=str(Path(td)/'transformer_60.pth')
        restored=restore(resume,model,optimizer,scheduler,scaler)
        assert restored['actual_iterations']==restored['optimizer_updates']==1
        # Fresh disabled model and missing text path: inference must still work.
        new=make_model(off,500,camera_num=7,view_num=3).to(args.device).eval()
        load_deployment(new,deploy)
        with torch.no_grad():
            new.neck_feat='before';torch.testing.assert_close(new(x[0],mode=1),pre,atol=0,rtol=0)
            new.neck_feat='after';torch.testing.assert_close(new(x[0],mode=1),post,atol=0,rtol=0)
    result=dict(status='PASSED_SERVER_192_SYNTHETIC_PREFLIGHT' if args.formal else 'PASSED_LOCAL_CPU_CODE_CHECKS',
        code_commit=code_commit(),device=args.device,visual_images=3*n,features=[3*n,768],logits=[3*n,500],
        resources=resources,initial_state_sha256=initial_hash,disabled_init_and_rng_equal=True,
        fixed_input_visual_outputs_equal=True,BN_updates_per_forward=1,aux_BN_classifier_gradients_None=True,
        P0_text_frozen=True,gain_aux_gradient_present=True,strict_full_deployment_reload=True,
        inference_without_text=True,CE=ce.item(),Triplet=tri.item(),text_CE=aux.item(),
        global_norm_before_clip=norm.item(),peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        limitation='Synthetic model checks and one optimizer step; not training or retrieval performance')
    write_json(output/'implementation_check.json',result)
    write_json(output/'config_diff_vs_m5_student.json',config_diff(c))
    print('M2-5-DT CHECKS PASSED',result['status'],'images',3*n,flush=True)

def evaluate(args,c):
    from datasets import make_dataloader
    from model import make_model
    from processor import do_inference
    from utils.logger import setup_logger
    from utils.runtime import configure_cudnn,runtime_summary
    assert c.TEST.NECK_FEAT in ('after','before')
    out=Path(c.OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(out),if_train=False)
    configure_cudnn(c.SOLVER.CUDNN_BENCHMARK);logger.info(runtime_summary())
    _,_,val,nq,nclass,ncam,nview=make_dataloader(c)
    assert sum(nq)==6405 and sum(len(loader.dataset) for loader in val)==100014
    # Disable the training-only switch before construction: no text file/tower load.
    inference=c.clone();inference.M2_DIRECT_TEXT.ENABLED=False
    model=make_model(inference,nclass,ncam,nview)
    obj=load_deployment(model,c.TEST.WEIGHT)
    assert obj['metadata']['epoch']==60
    write_json(out/'evaluation_manifest.json',dict(epoch=60,readout=c.TEST.NECK_FEAT,
        checkpoint=str(Path(c.TEST.WEIGHT).resolve()),strict_state_loading=True,text_file_required=False,
        training_commit=obj['metadata']['code_commit'],resources=obj['metadata']['resources'],config=c.dump()))
    logger.info('DT strict epoch60 reload; readout=%s; no text or teacher at inference',c.TEST.NECK_FEAT)
    do_inference(c,model,val,nq)
    manifest=__import__('json').loads((out/'evaluation_manifest.json').read_text(encoding='utf-8'))
    manifest['status']='COMPLETED_FULL_EVALUATION';write_json(out/'evaluation_manifest.json',manifest)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['preflight','eval'],required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--images-per-modality',type=int,default=64)
    parser.add_argument('--formal',action='store_true')
    parser.add_argument('--output',default='work/dt_preflight')
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args=parser.parse_args();c=configure(args)
    if args.action=='preflight':preflight(args,c)
    else:evaluate(args,c)
