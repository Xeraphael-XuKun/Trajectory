"""Merge I1 configs, then enter the unchanged baseline training loop plus hooks."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description='ICLR-inspired student; this branch defaults to I3 M5-Paired')
    p.add_argument('--base-config')
    p.add_argument('--common-config')
    p.add_argument('--config', default=str(ROOT/'configs/iclr_exp/m5_paired.yml'))
    p.add_argument('--resources')
    p.add_argument('--resume', default='')
    p.add_argument('opts', nargs=argparse.REMAINDER)
    return p


def main():
    from tools.iclr_exp.configuration import load_config
    args = parser().parse_args()
    opts = list(args.opts)
    if args.resume:
        opts += ['ICLR_EXP.RESUME', args.resume]
    c = load_config(args.base_config, args.common_config, args.config, args.resources, opts)
    if not c.ICLR_EXP.RESUME and (Path(c.OUTPUT_DIR)/'resume_latest.pth').exists():
        raise ValueError('output already has an ICLR checkpoint; use --resume or a new output directory')
    from train import setup_cuda_visible_devices, set_seed
    setup_cuda_visible_devices(c)
    set_seed(c.SOLVER.SEED, c.SOLVER.CUDNN_BENCHMARK)
    from datasets import make_dataloader
    from model import make_model
    from loss import make_loss
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from processor import do_train
    from utils.logger import setup_logger
    from utils.runtime import runtime_summary
    os.makedirs(c.OUTPUT_DIR, exist_ok=True)
    log = setup_logger('transreid', c.OUTPUT_DIR, if_train=True)
    log.info(runtime_summary())
    loaders = make_dataloader(c)
    train_loader, _, val_loaders, queries, classes, cameras, views = loaders
    if classes != 500:
        raise ValueError('ICLR requires WHU-MARS-1000 with 500 training identities')
    model = make_model(c, classes, cameras, views)
    loss_fn, center = make_loss(c, classes)
    optimizer, opt_center = make_optimizer(c, model, center)
    scheduler = create_scheduler(c, optimizer, iters_per_epoch=len(train_loader))
    with open(os.path.join(c.OUTPUT_DIR, 'config_merged.yml'), 'w', encoding='utf-8') as f:
        f.write(c.dump())
    with open(os.path.join(c.OUTPUT_DIR, 'optimizer_roles.json'), 'w', encoding='utf-8') as f:
        json.dump(model.iclr_optimizer_roles, f, indent=2)
    log.info('%s: 60-epoch student; original identity loop; nominal loader length=%d; observed batches counted separately', c.ICLR_EXP.VARIANT, len(train_loader))
    do_train(c, model, center, train_loader, val_loaders, optimizer, opt_center,
             scheduler, loss_fn, queries, 0)


if __name__ == '__main__':
    main()
