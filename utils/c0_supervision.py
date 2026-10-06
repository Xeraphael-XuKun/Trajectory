"""Compact epoch diagnostics; no graphs or sample features saved to disk."""
import json
import math
from pathlib import Path
import torch


class C0AuxMeter:
    def __init__(self, variant):
        self.variant = variant
        self.total_checks = {}
        self.reset()

    def reset(self):
        self.probes = {}
        self.loss_sum = 0.0
        self.steps = 0

    def update(self, loss, statistics):
        layers = list(statistics)
        packed = [torch.cat((statistics[k][0].flatten(), statistics[k][1].reshape(1)))
                  for k in layers]
        rows = torch.stack(packed).float().cpu().tolist()
        self.loss_sum += float(loss.detach())
        self.steps += 1
        for layer, values in zip(layers, rows):
            self.total_checks[layer] = self.total_checks.get(layer, 0) + 1
            probe = self.probes.setdefault(layer, {'checks': 0, 'ratio': 0.0,
                                                   'groups': [[0.0] * 11 for _ in range(4)]})
            probe['checks'] += 1
            probe['ratio'] += values[-1]
            for index in range(4):
                row = values[index * 11:(index + 1) * 11]
                dst = probe['groups'][index]
                first = dst[1] == 0
                for j in range(8):
                    dst[j] += row[j]
                if row[1]:
                    dst[8] = row[8] if first else min(dst[8], row[8])
                    dst[9] = row[9] if first else max(dst[9], row[9])
                dst[10] += row[10]

    def write(self, output_dir, logger, epoch, epoch_seconds, iterations, peak_bytes):
        summary = {'epoch': epoch, 'variant': self.variant, 'aux_steps': self.steps,
                   'aux_loss': self.loss_sum / max(self.steps, 1),
                   'epoch_train_seconds': epoch_seconds,
                   'seconds_per_iteration': epoch_seconds / max(iterations, 1),
                   'peak_allocated_gib': peak_bytes / 1024 ** 3, 'probes': {}}
        for layer, probe in self.probes.items():
            groups = {}
            for index, row in enumerate(probe['groups']):
                a, n, d0, d1, delta, square, harmed, active, low, high, loss = row
                denom = max(n, 1)
                groups['g' + str(index)] = {
                    'anchor_group_count': int(a), 'pair_count': int(n),
                    'd0_mean': d0 / denom, 'd1_mean': d1 / denom,
                    'delta_mean': delta / denom,
                    'delta_std': math.sqrt(max(0.0, square / denom - (delta / denom) ** 2)),
                    'delta_min': low, 'delta_max': high,
                    'harmed_fraction': harmed / denom, 'active_fraction': active / denom,
                    'loss_mean_per_check': loss / probe['checks'],
                }
            name = 'tail_10_12' if self.variant == 'c2' else 'block_' + str(layer)
            summary['probes'][name] = {
                'checks': probe['checks'], 'checks_total': self.total_checks[layer],
                'correction_input_norm_ratio': probe['ratio'] / probe['checks'], 'groups': groups}
        encoded = json.dumps(summary, ensure_ascii=False, allow_nan=False)
        logger.info('C0Aux ' + encoded)
        with (Path(output_dir) / 'c0_aux_stats.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
