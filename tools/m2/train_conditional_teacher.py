"""M2-5 stage B reuses the original loader/loss/optimizer/scheduler/processor."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.m2.artifacts import load_config, seed_stage

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('opts', nargs=argparse.REMAINDER)
    a = p.parse_args()
    c = load_config(a.config_file, a.opts)
    c.M2.TRAIN_STAGE = 'teacher'
    c.MODEL.TOKEN_TRAJECTORY = False
    c.MODEL.PRETRAIN_CHOICE = 'imagenet'
    c.SOLVER.MAX_EPOCHS = c.TEACHER_STAGE.MAX_EPOCHS
    c.OUTPUT_DIR = c.TEACHER_STAGE.OUTPUT_DIR
    c.freeze()
    seed_stage(c)
    from datasets import make_dataloader
    from model.m2_teacher import ConditionalVisualTeacher
    from loss import make_loss
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from processor import do_train
    from utils.logger import setup_logger
    Path(c.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    setup_logger('transreid', c.OUTPUT_DIR, if_train=True)
    Path(c.OUTPUT_DIR, 'config_merged.yml').write_text(c.dump(), encoding='utf-8')
    train_loader, _, val_loaders, queries, classes, cameras, views = make_dataloader(c)
    model = ConditionalVisualTeacher(c, classes, cameras, views)
    loss, center = make_loss(c, classes)
    opt, opt_center = make_optimizer(c, model, center)
    scheduler = create_scheduler(c, opt, iters_per_epoch=len(train_loader))
    do_train(c, model, center, train_loader, val_loaders, opt, opt_center, scheduler, loss, queries, 0)

if __name__ == '__main__':
    main()
