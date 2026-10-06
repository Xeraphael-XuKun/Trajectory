"""可选服务器预检：真实 dataloader 一批图像、AMP forward/backward，不启动训练。"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.cuda import amp
from config import cfg
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from loss.m2_losses import m2_a_loss, m2_b_loss
from solver import make_optimizer
from train import set_seed
from utils.runtime import runtime_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",required=True)
    parser.add_argument("opts",nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if cfg.M2.MODE == "none":
        raise ValueError("此预检只用于 M2 配置")
    set_seed(cfg.SOLVER.SEED,cfg.SOLVER.CUDNN_BENCHMARK)
    print(runtime_summary())
    train_loader, _, _, _, nclasses, ncams, nviews = make_dataloader(cfg)
    imgs,pid,cams = next(iter(train_loader))
    imgs=[x.cuda() for x in imgs]; pid=pid.cuda(); cams=[x.cuda() for x in cams]
    model=make_model(cfg,nclasses,ncams,nviews).cuda().train()
    loss_fn,center=make_loss(cfg,nclasses)
    optimizer,_=make_optimizer(cfg,model,center)
    model.set_m2_epoch(6)
    tracked=int(model.bottleneck.num_batches_tracked)
    target=pid.repeat(len(imgs))
    scaler=amp.GradScaler()
    with amp.autocast():
        out=model(imgs,pid,cams)
        main,_,_=loss_fn(out[0],out[1],target,target_ce=None)
        aux=out[3]
        extra=main.new_zeros(())
        stats={}
        if aux["active"]:
            if cfg.M2.MODE=="a":
                extra,stats=m2_a_loss(aux["after"],aux["before"],target,
                                     torch.cat(cams),aux["modality"],cfg.M2)
            else:
                extra,stats=m2_b_loss(out[2],target,torch.cat(cams),aux["modality"],cfg.M2)
        total=main+cfg.M2.WEIGHT*extra
    assert int(model.bottleneck.num_batches_tracked)==tracked+1
    if cfg.M2.MODE=="a" and aux["active"]:
        params=list(model.named_parameters())
        aux_grads=torch.autograd.grad(extra,[p for n,p in params if p.requires_grad],
                                      allow_unused=True,retain_graph=True)
        names=[n for n,p in params if p.requires_grad]
        assert all(g is None for n,g in zip(names,aux_grads) if not n.startswith("m2_reader."))
    assert torch.isfinite(total)
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    scaler.step(optimizer); scaler.update()
    print(json.dumps(dict(status="M2_REAL_BATCH_AMP_ONE_UPDATE_OK",mode=cfg.M2.MODE,
          objective=cfg.M2.A_OBJECTIVE,weighting=cfg.M2.B_WEIGHTING,
          images=len(target),main_loss=float(main),aux_loss=float(extra),stats=stats),
          ensure_ascii=False))
    print("这是一批预检，不代表已启动或完成正式 60 epoch 训练。")
