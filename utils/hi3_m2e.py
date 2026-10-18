"""Exact first-round HI3 contract and passive epoch diagnostics."""
import json
from pathlib import Path
import torch
from loss.history_prediction import history_diagnostics


def validate_hi3(c):
    h = c.HISTORY
    if (not h.ENABLED or c.MODEL.TOKEN_TRAJECTORY or h.RANK != 64 or h.ALPHA != .25
            or h.TOKEN_SCOPE != 'patch' or h.GAIN_MODE != 'linear'
            or list(h.PREDICTION_LAYERS) != list(range(2, 13)) or h.LOSS_WEIGHT != 1.
            or h.INIT_SEED != 1235 or h.WRITE_WEIGHT_DECAY != -1.
            or h.STOP_AFTER_EPOCH != 0 or h.GRAD_DIAG_PERIOD != 0
            or c.SOLVER.WEIGHT_DECAY != 1e-4 or c.SOLVER.WEIGHT_DECAY_BIAS != 1e-4):
        raise ValueError('This combination preserves first-round HI3, not HF/no-WD/CLS variants')


class HistoryEpochMonitor:
    def __init__(self, cfg, adapter):
        self.cfg, self.adapter = cfg, adapter
        self.reset()

    def reset(self):
        self.count = self.samples = 0
        self.sums = dict(ce=0., triplet=0., prediction_loss=0., kd_loss=0.)
        self.layers = {}

    @torch.no_grad()
    def observe(self, aux, ce, triplet, pred, kd, count, sampled):
        self.count += count
        for key, value in zip(self.sums, (ce, triplet, pred, kd)):
            self.sums[key] += float(value.detach()) * count
        if sampled:
            self.samples += 1
            for layer, values in history_diagnostics(aux).items():
                sums = self.layers.setdefault(layer, {})
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.) + value

    @torch.no_grad()
    def finish(self, epoch, iterations, total_iterations, updates, skips, logger):
        values = {k: v / self.count for k, v in self.sums.items()}
        layers = {l: {k: v / self.samples for k, v in row.items()} for l, row in self.layers.items()}
        for row in layers.values():
            if 'copy_mse' in row:
                row['prediction_gain_ratio_of_means'] = 1. - row['prediction_mse'] / max(row['copy_mse'], 1e-8)
        families = {}
        for name in ('velocity_gain', 'innovation_gain', 'predictors'):
            tensors = [p for key, p in self.adapter.named_parameters() if key.split('.')[0] == name]
            families[name] = (sum(p.detach().double().square().sum().item() for p in tensors)
                              / sum(p.numel() for p in tensors)) ** .5
        record = dict(epoch=epoch, iterations=iterations, total_iterations=total_iterations,
                      optimizer_updates=updates, amp_skipped_steps_total=skips,
                      diagnostic_batches=self.samples, layers=layers, parameters_rms=families,
                      gain_distribution=self.adapter.gain_diagnostics(), **values)
        record['weighted_prediction_loss'] = self.cfg.HISTORY.LOSS_WEIGHT * values['prediction_loss']
        record['weighted_kd_loss'] = self.cfg.M2.KD_WEIGHT * values['kd_loss'] if self.cfg.M2.ENABLED else 0.
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (Path(self.cfg.OUTPUT_DIR) / 'history_epoch.jsonl').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
        logger.info('History P3+M2-e: ' + line)
