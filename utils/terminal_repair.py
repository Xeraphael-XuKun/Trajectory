"""Epoch diagnostics for terminal repair; does not change optimization."""
import json
import math
from pathlib import Path


class TerminalRepairMeter:
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.steps = self.active_steps = self.valid_steps = self.skips = 0
        self.ce = self.triplet = self.raw = 0.0
        self.anchors = self.pairs = 0
        self.ess = self.off = self.on = 0.0
        self.grad_sum = self.grad_max = 0.0
        self.finite_grad_steps = self.clipped_steps = self.nonfinite_grad_steps = 0

    def update(self, ce, triplet, raw, stats, grad_norm, amp_skipped):
        self.steps += 1
        self.ce += float(ce)
        self.triplet += float(triplet)
        self.skips += int(amp_skipped)
        norm = float(grad_norm)
        if math.isfinite(norm):
            self.grad_sum += norm
            self.grad_max = max(self.grad_max, norm)
            self.finite_grad_steps += 1
            self.clipped_steps += int(norm > 1.0)
        else:
            self.nonfinite_grad_steps += 1
        if raw is None:
            return
        self.active_steps += 1
        self.raw += float(raw.detach())
        a, p = stats['valid_anchors'], stats['positive_pairs']
        self.anchors += a
        self.pairs += p
        if a:
            self.valid_steps += 1
            self.ess += stats['ess_fraction'] * a
            self.off += stats['off_soft_error'] * p
            self.on += stats['on_soft_error'] * p

    def write(self, output_dir, logger, epoch, seconds, peak_bytes):
        r = self.settings
        ce, triplet = self.ce / max(1, self.steps), self.triplet / max(1, self.steps)
        raw = self.raw / self.active_steps if self.active_steps else None
        weighted = raw * r.WEIGHT if raw is not None else None
        row = dict(epoch=epoch, weighting=r.WEIGHTING, gradient_scope=r.GRADIENT_SCOPE,
                   configured_weight=r.WEIGHT, temperature=r.TEMPERATURE,
                   first_active_epoch=r.WARMUP_EPOCHS + 1,
                   applied_weight=r.WEIGHT if epoch > r.WARMUP_EPOCHS else 0.0,
                   iterations=self.steps, auxiliary_iterations=self.active_steps,
                   valid_iterations=self.valid_steps, ce=ce, triplet=triplet,
                   repair_raw=raw, repair_weighted=weighted,
                   weighted_to_main_sum=(weighted / (ce + triplet)
                                         if weighted is not None and ce + triplet > 0 else None),
                   valid_anchors=self.anchors, positive_pairs=self.pairs,
                   ess_fraction=self.ess / self.anchors if self.anchors else None,
                   off_soft_error=self.off / self.pairs if self.pairs else None,
                   on_soft_error=self.on / self.pairs if self.pairs else None,
                   grad_norm_mean=self.grad_sum / self.finite_grad_steps if self.finite_grad_steps else None,
                   grad_norm_max=self.grad_max if self.finite_grad_steps else None,
                   clipped_steps=self.clipped_steps, nonfinite_grad_steps=self.nonfinite_grad_steps,
                   amp_skipped_steps=self.skips, seconds=seconds, peak_allocated_bytes=peak_bytes)
        with (Path(output_dir) / 'terminal_repair_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        logger.info('TerminalRepair %s', json.dumps(row, allow_nan=False))
