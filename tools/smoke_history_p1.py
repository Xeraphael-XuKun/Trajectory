"""One-batch P1 AMP trainer preflight, preserving the original 60-epoch schedule."""
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
from smoke_history import ShortSteps


def main(args):
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=False)
    c = cfg.clone(); c.merge_from_file(str(ROOT / 'configs/history_P1.yml'))
    c.MODEL.PRETRAIN_PATH = args.pretrained
    c.DATASETS.ROOT_DIR = args.data_root
    c.HISTORY.STOP_AFTER_EPOCH = 1
    c.HISTORY.GRAD_DIAG_PERIOD = 4
    c.SOLVER.EVAL_PERIOD = 999
    c.SOLVER.LOG_PERIOD = 4
    c.SOLVER.IMS_PER_BATCH = args.batch_size
    c.DATALOADER.NUM_INSTANCE = args.num_instance
    c.DATALOADER.NUM_WORKERS = 0
    c.OUTPUT_DIR = str(output)
    c.freeze()
    set_seed(c.SOLVER.SEED, c.SOLVER.CUDNN_BENCHMARK)
    logger = setup_logger('transreid', str(output), if_train=True)
    if args.verify_only:
        logger.info('VERIFY_EXISTING_PREFLIGHT: load saved weights and logs; no optimizer updates')
    else:
        logger.info('P1_PREFLIGHT_ONLY: one real batch repeated for %d attempts; not a formal epoch', args.steps)
    logger.info(runtime_summary())
    (output / 'resolved_config.yml').write_text(c.dump(), encoding='utf-8')
    loader, _, _, _, classes, cameras, views = make_dataloader(c)
    batch = next(iter(loader))
    short_loader = ShortSteps(batch, args.steps)
    model = make_model(c, classes, cameras, views)
    loss, center = make_loss(c, classes)
    optimizer, optimizer_center = make_optimizer(c, model, center)
    # Keep the actual sampler's iteration scale, not the repeated-batch loop length.
    scheduler = create_scheduler(c, optimizer, iters_per_epoch=len(loader))
    if args.verify_only:
        model.cuda()
        model.load_state_dict(torch.load(str(output / 'transformer_1.pth'), map_location='cpu'), strict=True)
    else:
        do_train(c, model, center, short_loader, {}, optimizer, optimizer_center,
                 scheduler, loss, {}, 0)
    epoch = json.loads((output / 'history_epoch.jsonl').read_text().splitlines()[-1])
    gradients = [json.loads(line) for line in (output / 'history_gradients.jsonl').read_text().splitlines()]
    assert args.steps - epoch['amp_skipped_steps'] >= 2
    assert epoch['gain_rms'] > 0 and epoch['layers']['1']['correction_rms'] == 0
    assert epoch['gradient_clipping'] == 'task_prediction_separate_1'
    assert all(row['identity_grad_norm']['predictors'] == 0 for row in gradients)
    assert all(row['prediction_raw_grad_norm']['gain'] == 0 and row['prediction_raw_grad_norm']['anchors'] == 0
               and row['prediction_raw_grad_norm']['correctors'] == 0 for row in gradients)
    # Early AMP-overflow samples are recorded as None by the existing monitor.
    assert any((row['identity_grad_norm']['correctors'] or 0) > 0 for row in gradients)
    assert all('separate_grad_preclip_norm' in row for row in gradients)
    model.eval(); image = batch[0][0].cuda()
    with torch.no_grad(): before = model(image, mode=1)
    # Strict same-architecture reload includes all tensors, including the classifier.
    model.load_state_dict(torch.load(str(output / 'transformer_1.pth'), map_location='cpu'), strict=True)
    with torch.no_grad(): after = model(image, mode=1)
    assert torch.equal(before, after)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report = {'status': 'PREFLIGHT_ONLY', 'runtime': runtime_summary(),
              'actual_images': sum(len(x) for x in batch[0]), 'logical_batch': args.batch_size,
              'num_instance': args.num_instance, 'schedule_epochs': c.SOLVER.MAX_EPOCHS,
              'schedule_iterations_per_epoch': len(loader), 'optimizer_attempts': args.steps,
              'amp_skipped_steps': epoch['amp_skipped_steps'],
              'optimizer_updates': args.steps - epoch['amp_skipped_steps'],
              'strict_disk_reload_equal': True, 'separate_gradients_verified': True,
              'verification_only': args.verify_only}
    peak_key = 'verification_peak_allocated_gib' if args.verify_only else 'training_peak_allocated_gib'
    report[peak_key] = torch.cuda.max_memory_allocated() / 2**30
    (output / 'preflight_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('P1_PREFLIGHT_OK ' + json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-instance', type=int, default=2)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--verify-only', action='store_true', help='verify an existing preflight without repeating its training')
    main(p.parse_args())
