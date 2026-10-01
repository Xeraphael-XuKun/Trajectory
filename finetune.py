"""本轮单卡正式训练入口。"""
import argparse
from config import load_config
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from processor import do_train
from utils.logger import setup_logger
from utils.runtime import prepare_run, runtime_summary


def main():
    parser = argparse.ArgumentParser(description='Baseline / Trajectory 单卡训练')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config_file, args.opts)
    cfg.freeze()
    prepare_run(cfg, training=True)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=True)
    logger.info(runtime_summary())
    logger.info('Running with config:\n%s', cfg)
    train_loader, val_loaders, num_querys, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_classes)
    loss_fn = make_loss(cfg, num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer, iters_per_epoch=len(train_loader))
    do_train(cfg, model, train_loader, val_loaders, optimizer, scheduler, loss_fn, num_querys)


if __name__ == '__main__':
    main()
