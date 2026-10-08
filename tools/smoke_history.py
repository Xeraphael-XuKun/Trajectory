"""Optional short real trainer smoke; not a formal experiment or tuning run."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from config import cfg
from train import set_seed
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from processor import do_train
from utils.logger import setup_logger
from utils.runtime import runtime_summary


class ShortSteps:
    def __init__(self, batch, steps):
        self.batch = batch
        self.steps = steps
        self.batch_size = len(batch[1])
    def __len__(self): return self.steps
    def __iter__(self):
        for _ in range(self.steps):
            yield self.batch


def main(args):
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    c = cfg.clone(); c.merge_from_file(str(ROOT / args.config))
    c.MODEL.PRETRAIN_PATH = args.pretrained
    c.DATASETS.ROOT_DIR = args.data_root
    c.SOLVER.MAX_EPOCHS = 1; c.SOLVER.EVAL_PERIOD = 999
    c.SOLVER.CHECKPOINT_PERIOD = 1; c.SOLVER.LOG_PERIOD = 1
    c.DATALOADER.NUM_WORKERS = 0
    c.DATALOADER.NUM_INSTANCE = 2; c.SOLVER.IMS_PER_BATCH = 4
    c.OUTPUT_DIR = str(output)
    c.freeze()
    set_seed(c.SOLVER.SEED, c.SOLVER.CUDNN_BENCHMARK)
    logger = setup_logger('transreid', str(output), if_train=True)
    logger.info('SMOKE ONLY: one real twelve-image batch repeated for %d optimizer attempts', args.steps)
    logger.info(runtime_summary())
    (output / 'resolved_config.yml').write_text(c.dump(), encoding='utf-8')
    loader, _, _, _, classes, cameras, views = make_dataloader(c)
    batch = next(iter(loader)); short_loader = ShortSteps(batch, args.steps)
    model = make_model(c, classes, cameras, views)
    loss, center = make_loss(c, classes)
    optimizer, optimizer_center = make_optimizer(c, model, center)
    scheduler = create_scheduler(c, optimizer, iters_per_epoch=len(short_loader))
    do_train(c, model, center, short_loader, {}, optimizer, optimizer_center,
             scheduler, loss, {}, 0)
    records = [json.loads(line) for line in (output/'history_epoch.jsonl').read_text().splitlines()]
    assert len(records) == 1 and records[0]['iterations'] == args.steps
    assert args.steps - records[0]['amp_skipped_steps'] >= 2, 'Too few successful optimizer updates'
    assert records[0]['gain_rms'] > 0
    model.eval(); image = batch[0][0].cuda()
    with torch.no_grad(): before_reload = model(image, mode=1)
    model.load_param(str(output/'transformer_1.pth'))
    with torch.no_grad(): after_reload = model(image, mode=1)
    assert torch.equal(before_reload, after_reload)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report = {'status': 'SMOKE_ONLY', 'runtime': runtime_summary(),
              'optimizer_attempts': args.steps, 'amp_skipped_steps': records[0]['amp_skipped_steps'],
              'disk_reload_equal': True, 'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
    (output/'smoke_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('HISTORY_TRAINER_SMOKE_OK ' + json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--config', default='configs/history_HI1.yml')
    p.add_argument('--pretrained', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    main(p.parse_args())
