import logging
import time
from pathlib import Path
import torch
from torch.cuda import amp
from utils.metrics import R1_mAP_eval, cmc_at
from utils.meter import AverageMeter
from utils.runtime import runtime_info, write_json
from loss.vtc import view_align


def evaluate(cfg, model, val_loaders, num_querys, logger):
    evaluator = R1_mAP_eval(
        max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING, top_k=cfg.TEST.TOP_K_EVAL,
        metric=cfg.TEST.METRIC, aerial_cams=cfg.DATASETS.AERIAL_CAMS, logger=logger)
    model.eval()
    with torch.no_grad():
        for mode, (loader, nq) in enumerate(zip(val_loaders, num_querys), start=1):
            evaluator.set_query_num(mode, nq)
            for img, pid, camid, camidt, modid, _ in loader:
                feat = model(img.cuda(), camids=camidt.cuda(), mode=mode)
                evaluator.update((feat, pid, camid, modid), mode)
    evaluator.split_all()
    cmc, mAP, pairs, q_pids, _, _, _, views = evaluator.compute()

    def metrics(c, m):
        return {'mAP': float(m * 100), 'Rank-1': cmc_at(c, 1) * 100,
                'Rank-5': cmc_at(c, 5) * 100, 'Rank-10': cmc_at(c, 10) * 100}

    result = {'overall': metrics(cmc, mAP), 'query_count': len(q_pids),
              'modality_pairs': [], 'view_pairs': []}
    for destination, entries in [('modality_pairs', pairs), ('view_pairs', views)]:
        for q, g, c, m, inp in entries:
            row = {'query': int(q) if destination == 'modality_pairs' else q,
                   'gallery': int(g) if destination == 'modality_pairs' else g,
                   **metrics(c, m), 'mINP': float(inp * 100)}
            result[destination].append(row)
    logger.info('Readout: %s; mAP: %.2f%%; Rank-1: %.2f%%; Rank-5: %.2f%%; Rank-10: %.2f%%',
                cfg.TEST.NECK_FEAT, *[result['overall'][k] for k in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']])
    return result


def do_train(cfg, model, train_loader, val_loaders, optimizer, scheduler, loss_fn, num_querys):
    logger = logging.getLogger('transreid.train')
    write_json(Path(cfg.OUTPUT_DIR) / 'training_summary.json', {
        'complete': False, 'experiment_id': cfg.EXPERIMENT.ID, 'epochs_completed': 0})
    model.cuda()
    scaler = amp.GradScaler()
    iterations, skipped = 0, 0
    text_weight = cfg.SOLVER.TEXT_LOSS_WEIGHT if cfg.MODEL.TEXT_ALIGN else 0.0
    logger.info('开始训练：%d epoch 主读出为 pre-BN，VTC=%s，AMP 初始 scale=%s',
                cfg.SOLVER.MAX_EPOCHS, cfg.MODEL.TEXT_ALIGN, scaler.get_scale())
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        start = time.time()
        meter, ce_meter, tri_meter = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        scheduler.step(epoch)
        epoch_batches = 0
        for n_iter, (imgs, target, camids) in enumerate(train_loader):
            scheduler.step_update((epoch - 1) * len(train_loader) + n_iter)
            optimizer.zero_grad()
            imgs = [img.cuda() for img in imgs]
            camids = [cam.cuda() for cam in camids]
            target = target.cuda()
            labels = target.repeat(len(imgs))
            with amp.autocast(enabled=True):
                out = model(imgs, target, camids)
                score, global_feat = out[0], out[1]
                loss, ce, triplet = loss_fn(score, global_feat, labels)
                if text_weight > 0:
                    vtc_loss, _ = view_align(out[3])
                    loss = loss + text_weight * vtc_loss
            if not torch.isfinite(loss):
                raise RuntimeError('loss 非有限值：epoch={}, iteration={}'.format(epoch, n_iter + 1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            iterations += 1
            epoch_batches += 1
            meter.update(loss.item(), target.shape[0])
            ce_meter.update(ce.item(), labels.shape[0])
            tri_meter.update(triplet.item(), labels.shape[0])
            if (n_iter + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                lrs = [g['lr'] for g in optimizer.param_groups]
                logger.info('Epoch[%d] Iteration[%d/%d] Loss=%.4f CE=%.4f Triplet=%.4f LR=%.2e/%.2e AMP skips=%d',
                            epoch, n_iter + 1, len(train_loader), meter.avg, ce_meter.avg,
                            tri_meter.avg, min(lrs), max(lrs), skipped)
        torch.cuda.synchronize()
        if epoch_batches == 0:
            raise RuntimeError('训练 epoch 为空')
        logger.info('Epoch %d done, batches=%d time=%.1fs', epoch, epoch_batches, time.time() - start)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            torch.save(model.state_dict(), Path(cfg.OUTPUT_DIR) / '{}_{}.pth'.format(cfg.MODEL.NAME, epoch))
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('训练期 pre-BN 监测，Epoch=%d', epoch)
            result = evaluate(cfg, model, val_loaders, num_querys, logger)
            write_json(Path(cfg.OUTPUT_DIR) / 'monitor_epoch_{}.json'.format(epoch), result)
            torch.cuda.empty_cache()
    write_json(Path(cfg.OUTPUT_DIR) / 'training_summary.json', {
        'complete': True, 'experiment_id': cfg.EXPERIMENT.ID,
        'epochs_completed': epoch, 'iterations': iterations,
        'optimizer_updates': iterations - skipped, 'amp_skips': skipped,
        'checkpoint': str((Path(cfg.OUTPUT_DIR) / '{}_{}.pth'.format(cfg.MODEL.NAME, epoch)).resolve()),
        'runtime': runtime_info()})


def do_inference(cfg, model, val_loaders, num_querys):
    logger = logging.getLogger('transreid.test')
    model.cuda()
    result = evaluate(cfg, model, val_loaders, num_querys, logger)
    result.update({
        'experiment_id': cfg.EXPERIMENT.ID, 'environment': cfg.EXPERIMENT.ENVIRONMENT,
        'normalization': cfg.EXPERIMENT.NORMALIZATION,
        'trajectory': cfg.MODEL.TOKEN_TRAJECTORY, 'vtc': cfg.MODEL.TEXT_ALIGN,
        'neck_feat': cfg.TEST.NECK_FEAT, 'seed': cfg.SOLVER.SEED,
        'checkpoint': str(Path(cfg.TEST.WEIGHT).resolve()),
        'pixel_mean': list(cfg.INPUT.PIXEL_MEAN), 'pixel_std': list(cfg.INPUT.PIXEL_STD),
        'config': cfg.dump(), 'runtime': runtime_info()})
    write_json(Path(cfg.OUTPUT_DIR) / 'metrics.json', result)
    return result
