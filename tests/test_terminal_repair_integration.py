"""Production routing, historical recipe isolation and auxiliary gradient checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from loss.terminal_repair import terminal_repair
from model.terminal_repair import validate_terminal_repair, readonly_neck
from test_c0_supervision import ROOT, tiny, wrapper, metadata

GROUPS = [('S1', 'uniform', 'joint', 1.0), ('S2', 'reference', 'joint', 1.0),
          ('S3', 'self', 'joint', 1.0), ('S4', 'reference', 'gain_only', 1.0),
          ('S5', 'reference', 'joint', 5.0)]


def settings(group):
    c = cfg.clone()
    c.merge_from_file(str(ROOT / ('configs/terminal_repair_' + group + '.yml')))
    return c


class TerminalIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_config_isolation_and_mapping(self):
        original = cfg.clone()
        original.merge_from_file(str(ROOT / 'configs/trajectory_C0.yml'))
        for group, weighting, scope, weight in GROUPS:
            c = settings(group)
            validate_terminal_repair(c)
            r = c.TERMINAL_REPAIR
            self.assertEqual((r.WEIGHTING, r.GRADIENT_SCOPE, r.WEIGHT), (weighting, scope, weight))
            self.assertEqual((r.ENABLED, r.TEMPERATURE, r.WARMUP_EPOCHS), (True, .07, 5))
            restored = c.clone()
            restored.TERMINAL_REPAIR = original.TERMINAL_REPAIR.clone()
            for owner, key in [('MODEL', 'PRETRAIN_PATH'), ('MODEL', 'TEXT_CLIP_PATH'),
                               ('DATASETS', 'ROOT_DIR'), ('TEST', 'NECK_FEAT')]:
                setattr(getattr(restored, owner), key, getattr(getattr(original, owner), key))
            restored.OUTPUT_DIR = original.OUTPUT_DIR
            self.assertEqual(restored.dump(), original.dump())
        for owner, key, value in [('C0_AUX', 'ENABLED', True), ('MODEL', 'VPR', True),
                                   ('MODEL', 'TOKEN_TRAJECTORY_ACCEL_MIX', 1.0),
                                   ('TERMINAL_REPAIR', 'GRADIENT_SCOPE', 'misspelled')]:
            c = settings('S2')
            setattr(getattr(c, owner), key, value)
            with self.assertRaises(ValueError):
                validate_terminal_repair(c)

    def test_forward_rng_bn_gradients_and_recurrence(self):
        original = wrapper(tiny())
        original.base.token_trajectory.gain.data.normal_(std=.04)
        images = [torch.randn(8, 3, 32, 16) for _ in range(3)]
        ids, cams, mods = metadata()
        rng = torch.random.get_rng_state()
        expected = original(images, ids[:8], list(cams.split(8)))
        after_rng = torch.random.get_rng_state()
        expected_neck = readonly_neck(expected[1], original.bottleneck)
        direction = torch.randn_like(expected_neck)
        expected_grad = torch.autograd.grad((expected_neck * direction).sum(),
                                            original.base.token_trajectory.gain, retain_graph=True)[0]
        # Restore the pre-forward BN state for each independent model.
        initial = copy.deepcopy(original.state_dict())
        initial['bottleneck.running_mean'].zero_()
        initial['bottleneck.running_var'].fill_(1)
        initial['bottleneck.num_batches_tracked'].zero_()
        joint_aux_grad = None
        for group, weighting, scope, weight in GROUPS:
            model = wrapper(tiny())
            model.load_state_dict(initial)
            model.terminal_repair_enabled = True
            model.terminal_repair_scope = scope
            torch.random.set_rng_state(rng)
            out = model(images, ids[:8], list(cams.split(8)), terminal_repair_active=True)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), after_rng))
            for a, b in zip(out[:3], expected):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(model.bottleneck.num_batches_tracked.item(), 1)
            self.assertTrue(torch.equal(out[3]['corrected'], expected_neck))
            self.assertFalse(out[3]['reference'].requires_grad)
            self.assertEqual(list(model.state_dict()), list(original.state_dict()))
            # S4 must preserve the complete gain derivative, not only output values.
            derivative = torch.autograd.grad((out[3]['corrected'] * direction).sum(),
                                             model.base.token_trajectory.gain, retain_graph=True)[0]
            torch.testing.assert_close(derivative, expected_grad, rtol=0, atol=0)
            loss, stats = terminal_repair(out[3]['reference'], out[3]['corrected'], ids, cams, mods,
                                          weighting=weighting)
            self.assertEqual(stats['valid_anchors'], 24)
            (weight * loss).backward()
            gain_grad = model.base.token_trajectory.gain.grad
            self.assertTrue((gain_grad.flatten(1).abs().sum(1) > 0).all())
            if group == 'S2':
                joint_aux_grad = gain_grad.clone()
            if group == 'S4':
                torch.testing.assert_close(gain_grad, joint_aux_grad, rtol=0, atol=0)
            if group == 'S5':
                relative_error = (gain_grad - 5 * joint_aux_grad).norm() / (5 * joint_aux_grad).norm()
                self.assertLess(relative_error.item(), 2e-5)
            self.assertIsNone(model.classifier.weight.grad)
            self.assertIsNone(model.bottleneck.weight.grad)
            backbone = [p for n, p in model.base.named_parameters() if not n.startswith('token_trajectory.')]
            if scope == 'gain_only':
                self.assertTrue(all(p.grad is None for p in backbone))
            else:
                self.assertGreater(sum(p.grad.abs().sum().item() for p in backbone if p.grad is not None), 0)
            # Inference ignores the training-only flag and preserves both C0 readouts.
            model.eval()
            for neck in ('before', 'after'):
                model.neck_feat = neck
                with torch.no_grad():
                    a = model(images[0], mode=1)
                    b = model(images[0], mode=1, terminal_repair_active=True)
                self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA production processor check')
    def test_production_warmup_fixed_weight_and_disk_reload(self):
        from processor import do_train
        from solver import make_optimizer
        from solver.scheduler_factory import create_scheduler
        from loss.triplet_loss import TripletLoss
        ids, cams, _ = metadata()
        images = [torch.randn(8, 3, 32, 16) for _ in range(3)]

        class Batches:
            batch_size = 8

            def __len__(self):
                return 2  # actual 1; diagnostics must count observed iterations

            def __iter__(self):
                yield images, ids[:8], list(cams.split(8))

        for group, _, scope, weight in GROUPS:
            with tempfile.TemporaryDirectory(dir=str(ROOT / 'work')) as directory:
                c = settings(group)
                c.SOLVER.MAX_EPOCHS = 7
                c.SOLVER.EVAL_PERIOD = c.SOLVER.LOG_PERIOD = 100
                c.SOLVER.CHECKPOINT_PERIOD = 7
                c.OUTPUT_DIR = directory
                model = wrapper(tiny()).cuda()
                model.terminal_repair_enabled = True
                model.terminal_repair_scope = scope
                center = nn.Linear(1, 1).cuda()
                optimizer, optimizer_center = make_optimizer(c, model, center)
                scheduler = create_scheduler(c, optimizer, iters_per_epoch=len(Batches()))

                def loss_fn(scores, features, labels, target_ce=None):
                    ce = F.cross_entropy(scores, labels)
                    tri = TripletLoss()(features, labels)[0]
                    return ce + tri, ce, tri

                do_train(c, model, center, Batches(), [], optimizer, optimizer_center,
                         scheduler, loss_fn, [], 0)
                rows = [json.loads(s) for s in (Path(directory) / 'terminal_repair_metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['iterations'] for r in rows], [1] * 7)
                self.assertEqual([r['auxiliary_iterations'] for r in rows], [0] * 5 + [1, 1])
                self.assertEqual([r['applied_weight'] for r in rows], [0] * 5 + [weight, weight])
                for row in rows[5:]:
                    self.assertEqual(row['repair_weighted'], weight * row['repair_raw'])
                    self.assertGreater(row['valid_anchors'], 0)
                restored = wrapper(tiny()).cuda()
                restored.load_param(str(Path(directory) / 'transformer_7.pth'))
                restored.eval()
                model.eval()
                for neck in ('before', 'after'):
                    model.neck_feat = restored.neck_feat = neck
                    with torch.no_grad():
                        a = model(images[0].cuda(), mode=1)
                        b = restored(images[0].cuda(), mode=1)
                    self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
