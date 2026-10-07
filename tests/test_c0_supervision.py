"""Mechanism checks for C0 auxiliary training, including historical path parity.

Run: python -m unittest discover -s tests -p test_c0_supervision.py -v
"""
import copy
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from model.backbones.vit_pytorch import TransReID
from model.make_model import build_transformer
from model.c0_supervision import (VARIANTS, validate_c0_aux, probe_layers,
                                  paired_probe, build_probes)
from loss.c0_supervision import layer_loss, c0_aux_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def tiny(factory=TransReID, enabled=True):
    return factory(img_size=(32, 16), patch_size=16, stride_size=16, embed_dim=8,
                   depth=12, num_heads=2, mlp_ratio=2, drop_path_rate=0.2,
                   drop_rate=0.1, attn_drop_rate=0.1,
                   token_trajectory=enabled, token_trajectory_accel_mix=0.0)


def wrapper(base, variant=None, factory=build_transformer):
    model = factory.__new__(factory)
    nn.Module.__init__(model)
    model.base = base
    model.classifier = nn.Linear(8, 2, bias=False)
    model.bottleneck = nn.BatchNorm1d(8)
    model.bottleneck.bias.requires_grad_(False)
    model.use_mod_delta = model.use_vpr = model.text_align = False
    model.c0_aux_variant = variant
    model.terminal_repair_enabled = False
    model.terminal_repair_scope = 'joint'
    model.neck_feat = 'after'
    return model


def metadata():
    # 2 IDs x 4 samples each x 3 modalities; both camera groups and
    # different ground cameras provide all four relation categories.
    ids = torch.tensor([0] * 4 + [1] * 4).repeat(3)
    cams = torch.tensor([0, 1, 5, 6] * 2).repeat(3)
    mods = torch.arange(3).repeat_interleave(8)
    return ids, cams, mods


def historical_module(path, name, package):
    source = subprocess.check_output(['git', 'show', '8fff926:' + path], cwd=ROOT).decode('utf-8')
    module = types.ModuleType(name)
    module.__package__ = package
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


class C0SupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_five_configs_are_c0_with_only_declared_changes(self):
        original = cfg.clone()
        original.merge_from_file(str(ROOT / 'configs/trajectory_C0.yml'))
        mapping = dict(zip(('R1', 'R2', 'R3', 'R4', 'R5'), VARIANTS))
        for group, variant in mapping.items():
            c = cfg.clone()
            c.merge_from_file(str(ROOT / ('configs/c0_aux_' + group + '.yml')))
            validate_c0_aux(c)
            self.assertEqual(c.C0_AUX.VARIANT, variant)
            restored = c.clone()
            restored.C0_AUX = original.C0_AUX.clone()
            for owner, key in [('MODEL', 'PRETRAIN_PATH'), ('MODEL', 'TEXT_CLIP_PATH'),
                               ('DATASETS', 'ROOT_DIR'), ('TEST', 'NECK_FEAT')]:
                setattr(getattr(restored, owner), key, getattr(getattr(original, owner), key))
            restored.OUTPUT_DIR = original.OUTPUT_DIR
            self.assertEqual(restored.dump(), original.dump())
        self.assertEqual([probe_layers('c1', i) for i in range(5)], [(7,), (9,), (11,), (7,), (9,)])
        self.assertEqual(probe_layers('c1_all3', 4), (7, 9, 11))
        self.assertEqual(probe_layers('c2', 4), (9,))

    def test_mixed_method_config_rejected(self):
        c = cfg.clone()
        c.merge_from_file(str(ROOT / 'configs/c0_aux_R1.yml'))
        for owner, field, value in [('MODEL', 'TOKEN_TRAJECTORY_ACCEL_MIX', 1.0),
                                    ('MODEL', 'VPR', True), ('SOLVER', 'LOSS_TYPE', 'pca'),
                                    ('C0_AUX', 'VARIANT', 'misspelled')]:
            d = c.clone()
            setattr(getattr(d, owner), field, value)
            with self.assertRaises(ValueError):
                validate_c0_aux(d)

    def test_checkpoint_reload_keeps_both_inference_readouts(self):
        model = wrapper(tiny(), 'c1_all3').eval()
        model.base.token_trajectory.gain.data.normal_(std=0.03)
        restored = wrapper(tiny(), 'c2').eval()
        image = torch.randn(4, 3, 32, 16)
        (ROOT / 'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str((ROOT / 'work').resolve())) as directory:
            checkpoint = Path(directory) / 'weights.pth'
            torch.save(model.state_dict(), checkpoint)
            restored.load_param(str(checkpoint))
            for readout in ('before', 'after'):
                model.neck_feat = restored.neck_feat = readout
                with torch.no_grad():
                    expected, actual = model(image, mode=1), restored(image, mode=1)
                self.assertTrue(torch.equal(expected, actual))

    def test_historical_baseline_c0_t0_forward_grad_update_rng_exact(self):
        old_vit = historical_module('model/backbones/vit_pytorch.py', 'old_vit', 'model.backbones')
        old_model = historical_module('model/make_model.py', 'old_model', 'model')
        images = [torch.randn(8, 3, 32, 16) for _ in range(3)]
        ids, cams, _ = metadata()
        for enabled, acceleration in ((False, 0.0), (True, 0.0), (True, 1.0)):
            torch.manual_seed(19)
            old = wrapper(tiny(old_vit.TransReID, enabled), factory=old_model.build_transformer)
            torch.manual_seed(19)
            new = wrapper(tiny(enabled=enabled))
            self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
            if enabled:
                old.base.token_trajectory.acceleration_mix = acceleration
                new.base.token_trajectory.acceleration_mix = acceleration
                old.base.token_trajectory.gain.data.normal_(std=0.02)
                new.load_state_dict(old.state_dict())
            opts = cfg.clone()
            center = nn.Linear(1, 1)
            opt_old, _ = make_optimizer(opts, old, center)
            opt_new, _ = make_optimizer(opts, new, center)
            outputs = []
            for model, opt in ((old, opt_old), (new, opt_new)):
                torch.manual_seed(101)
                out = model(images, ids[:8], list(cams.split(8)))
                rng = torch.random.get_rng_state()
                loss = F.cross_entropy(out[0], ids) + TripletLoss()(out[1], ids)[0]
                loss.backward()
                grads = {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}
                opt.step()
                outputs.append((out, rng, grads, copy.deepcopy(model.state_dict())))
            a, b = outputs
            for x, y in zip(a[0], b[0]):
                self.assertTrue(torch.equal(x, y))
            self.assertTrue(torch.equal(a[1], b[1]))
            for k in a[2]:
                self.assertTrue((a[2][k] is None and b[2][k] is None) or
                                torch.equal(a[2][k], b[2][k]), k)
            for k in a[3]:
                self.assertTrue(torch.equal(a[3][k], b[3][k]), k)

    def test_active_probes_preserve_main_outputs_bn_and_rng(self):
        images = [torch.randn(8, 3, 32, 16) for _ in range(3)]
        ids, cams, _ = metadata()
        for variant in VARIANTS:
            model = wrapper(tiny(), variant)
            ordinary = copy.deepcopy(model)
            torch.manual_seed(42)
            expected = ordinary(images, ids[:8], list(cams.split(8)))
            rng = torch.random.get_rng_state()
            torch.manual_seed(42)
            actual = model(images, ids[:8], list(cams.split(8)), c0_probe_step=0)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for a, b in zip(expected, actual[:3]):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(model.bottleneck.num_batches_tracked.item(), 1)
            self.assertTrue(torch.equal(model.bottleneck.running_var, ordinary.bottleneck.running_var))
            for z0, z1, ratio in actual[3]['probes'].values():
                self.assertTrue(torch.equal(z0, z1))
                self.assertEqual(ratio.item(), 0)
            # Auxiliary backward must not reach prefix, block, norm or BN weights.
            loss, _ = c0_aux_loss(actual[3], ids, cfg.C0_AUX)
            loss.backward()
            grad = model.base.token_trajectory.gain.grad
            selected = (9, 10, 11) if variant == 'c2' else probe_layers(variant, 0)
            for index in range(1, 12):
                if index in selected:
                    self.assertGreater(grad[index - 1].abs().sum().item(), 0)
                else:
                    self.assertEqual(grad[index - 1].abs().sum().item(), 0)
            self.assertGreater(grad[:, :, 1:].abs().sum().item(), 0)
            for name, p in model.named_parameters():
                if name != 'base.token_trajectory.gain':
                    self.assertIsNone(p.grad, name)

    def test_c2_nonzero_recurrence_matches_main_terminal_readout(self):
        base = tiny()
        base.token_trajectory.gain.data.normal_(std=0.04)
        bn = nn.BatchNorm1d(8)
        image = torch.randn(24, 3, 32, 16)
        final, states = base(image, c0_probe_layers=(9,))
        bn(final)  # The main forward's one and only BN update.
        expected = F.normalize(F.batch_norm(final, bn.running_mean, bn.running_var,
                                            bn.weight, bn.bias, training=False, eps=bn.eps), dim=-1)
        _, actual, _ = paired_probe(base, bn, states[9], 9, tail=True)
        self.assertTrue(torch.equal(expected, actual))
        # Distinct from detaching intermediate velocities: compare full tail derivative.
        direction = torch.randn_like(actual)
        grads = torch.autograd.grad((actual * direction).sum(), base.token_trajectory.gain)[0]
        for row in (8, 9, 10):
            self.assertGreater(grads[row].abs().sum().item(), 0)

    def test_loss_matches_explicit_relation_loop(self):
        ids, cams, mods = metadata()
        z0 = F.normalize(torch.randn(24, 8), dim=-1)
        z1 = F.normalize(z0 + 0.1 * torch.randn_like(z0), dim=-1).requires_grad_()
        s0, s1 = z0 @ z0.t(), z1 @ z1.t()
        settings = cfg.C0_AUX
        for variant in ('c1', 'c1_absolute', 'c1_pooled'):
            actual, stats = layer_loss(z0, z1, ids, cams, mods, settings, variant)
            grouped = [[] for _ in range(4)]
            counts = [0] * 4
            for i in range(24):
                sample = [[] for _ in range(4)]
                for p in range(24):
                    if ids[p] != ids[i] or cams[p] == cams[i]:
                        continue
                    negatives = [n for n in range(24) if ids[n] != ids[i] and cams[n] != cams[i]
                                 and mods[n] == mods[p] and (cams[n] >= 5) == (cams[p] >= 5)]
                    if not negatives:
                        continue
                    n = max(negatives, key=lambda n: float(s0[i, n]))
                    group = int(mods[p] != mods[i]) + 2 * int((cams[p] >= 5) != (cams[i] >= 5))
                    d0, d1 = s0[i, p] - s0[i, n], s1[i, p] - s1[i, n]
                    if group == 0:
                        value = F.relu(d0 - settings.KEEP_TOL - d1)
                    elif variant == 'c1_absolute':
                        value = F.relu(settings.MARGIN - d1)
                    else:
                        value = F.relu(d0 + (settings.MARGIN - d0).clamp(0, settings.MAX_GAIN) - d1)
                    sample[group].append(value)
                    counts[group] += 1
                for group in range(4):
                    if sample[group]:
                        grouped[group].append(torch.stack(sample[group]).mean())
            if variant == 'c1_pooled':
                cross = torch.stack(sum(grouped[1:], [])).mean()
            else:
                cross = torch.stack([torch.stack(g).mean() for g in grouped[1:] if g]).mean()
            expected = cross + settings.KEEP_WEIGHT * torch.stack(grouped[0]).mean()
            torch.testing.assert_close(actual, expected)
            self.assertEqual(stats[:, 1].tolist(), counts)
            ga = torch.autograd.grad(actual, z1, retain_graph=True)[0]
            gb = torch.autograd.grad(expected, z1, retain_graph=True)[0]
            torch.testing.assert_close(ga, gb)

    def test_missing_groups_zero_loss_and_all3_mean(self):
        base = tiny()
        x = torch.randn(8, 3, 32, 16)
        _, states = base(x, c0_probe_layers=(7, 9, 11))
        probes = build_probes(base, nn.BatchNorm1d(8), states, 'c1_all3')
        aux = {'probes': probes, 'cams': torch.zeros(8, dtype=torch.long),
               'mods': torch.arange(8) % 3}
        loss, stats = c0_aux_loss(aux, torch.arange(8), cfg.C0_AUX)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(base.token_trajectory.gain.grad.abs().sum().item(), 0)
        z0 = F.normalize(torch.randn(24, 8), dim=-1)
        ids, cams, mods = metadata()
        probes = {k: (z0, F.normalize(torch.randn(24, 8), dim=-1), torch.tensor(0.1))
                  for k in (7, 9, 11)}
        settings = cfg.C0_AUX.clone()
        settings.VARIANT = 'c1_all3'
        loss, _ = c0_aux_loss({'probes': probes, 'cams': cams, 'mods': mods}, ids, settings)
        expected = torch.stack([layer_loss(a, b, ids, cams, mods, settings, 'c1_all3')[0]
                                for a, b, _ in probes.values()]).mean()
        torch.testing.assert_close(loss, expected)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration check')
    def test_processor_warmup_and_cross_epoch_routing(self):
        from processor import do_train
        from solver.scheduler_factory import create_scheduler
        ids, cams, _ = metadata()
        images = [torch.randn(8, 3, 32, 16) for _ in range(3)]

        class FixedBatches:
            batch_size = 8

            def __len__(self):
                # PKM reports an estimate: actual iterations can be fewer.
                # Four advertised vs two yielded exposes epoch-phase jumps.
                return 4

            def __iter__(self):
                for _ in range(2):
                    yield images, ids[:8], list(cams.split(8))

        loader = FixedBatches()
        for variant in VARIANTS:
            # Keep auto-cleaned temporary output inside the known workspace.
            (ROOT / 'work').mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str((ROOT / 'work').resolve())) as directory:
                c = cfg.clone()
                c.merge_from_file(str(ROOT / 'configs/c0_aux_R1.yml'))
                c.C0_AUX.VARIANT = variant
                c.SOLVER.MAX_EPOCHS = 7
                c.SOLVER.EVAL_PERIOD = c.SOLVER.CHECKPOINT_PERIOD = c.SOLVER.LOG_PERIOD = 100
                c.OUTPUT_DIR = directory
                model = wrapper(tiny(), variant).cuda()
                center = nn.Linear(1, 1).cuda()
                optimizer, optimizer_center = make_optimizer(c, model, center)
                scheduler = create_scheduler(c, optimizer, iters_per_epoch=len(loader))

                def loss_fn(scores, features, labels, target_ce=None):
                    ce = F.cross_entropy(scores, labels)
                    tri = TripletLoss()(features, labels)[0]
                    return ce + tri, ce, tri

                do_train(c, model, center, loader, [], optimizer, optimizer_center,
                         scheduler, loss_fn, [], 0)
                reports = [json.loads(row) for row in (Path(directory) / 'c0_aux_stats.jsonl').read_text().splitlines()]
                self.assertEqual([r['aux_steps'] for r in reports], [0] * 5 + [2, 2])
                if variant == 'c2':
                    self.assertEqual(reports[-1]['probes']['tail_10_12']['checks_total'], 4)
                elif variant == 'c1_all3':
                    self.assertEqual([p['checks_total'] for p in reports[-1]['probes'].values()], [4, 4, 4])
                else:
                    self.assertEqual(list(reports[5]['probes']), ['block_8', 'block_10'])
                    self.assertEqual(list(reports[6]['probes']), ['block_12', 'block_8'])
                    self.assertEqual(reports[6]['probes']['block_8']['checks_total'], 2)


if __name__ == '__main__':
    unittest.main()
