"""Small real-data check of baseline M2-5 with the saved epoch60 teacher."""
import argparse
import gc
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import make_model
from model.m2_teacher import normalized_clip_logits, kd_kl
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from tools.m2.artifacts import load_config, seed_stage, train_source
from utils.m5_checkpoint import load_teacher, attach_metadata, save_training, restore_training


def check(c, device, artifacts_only=False):
    if c.MODEL.TOKEN_TRAJECTORY or c.M2.TRAIN_STAGE != 'student':
        raise ValueError('this check is for the baseline student with a reused teacher')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    seed_stage(c)
    teacher, payload = load_teacher(c, 500, device)
    bank = torch.load(c.M2.TEXT_BANK, map_location='cpu', weights_only=False)
    assert teacher.base.token_trajectory is None
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    print('REUSED_TEACHER_STRICT_OK: epoch=%d code_commit=%s; shared text/prompt/train/CLIP verified' % (
        payload['metadata']['epoch'], payload['metadata']['code_commit']), flush=True)
    if artifacts_only:
        return

    seed_stage(c)
    student = make_model(c, 500, 7, 0).to(device)
    student_rng = torch.get_rng_state().clone()
    off = c.clone()
    off.M2.ENABLED = False
    seed_stage(off)
    reference = make_model(off, 500, 7, 0).to(device)
    assert torch.equal(student_rng, torch.get_rng_state())
    assert student.base.token_trajectory is None
    assert not any('token_trajectory' in name for name in student.state_dict())
    assert student.state_dict().keys() == reference.state_dict().keys()
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    assert not student.base.clip_proj.requires_grad

    rows, _ = train_source(c)
    ids = sorted({row[1] for row in rows})[:3]
    transform = transforms.Compose([transforms.Resize(c.INPUT.SIZE_TEST), transforms.ToTensor(),
        transforms.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD)])
    images, cameras = [], []
    for modality in range(3):
        group, cams = [], []
        for y in ids:
            row = next(row for row in rows if row[1] == y and row[3] == modality)
            with Image.open(row[0]) as image:
                group.append(transform(image.convert('RGB')))
            cams.append(row[2])
        images.append(torch.stack(group).to(device))
        cameras.append(torch.tensor(cams, device=device))
    joined = torch.cat(images)
    student.eval()
    reference.eval()
    with torch.no_grad():
        torch.testing.assert_close(student(joined, mode=1), reference(joined, mode=1), rtol=0, atol=0)
    del reference
    gc.collect()
    print('BASELINE_STUDENT_OK: exact initial tensors/RNG/eval features; no C0 parameters', flush=True)

    target = torch.tensor(ids, device=device).repeat(3)
    modalities = torch.arange(3, device=device).repeat_interleave(len(ids))
    cams = torch.cat(cameras)
    platforms = ((cams == 5) | (cams == 6)).long()
    text = bank['text_bank'].to(device).detach()
    student.train()
    triplet = TripletLoss()
    # The diagnostic needs no unused GPU-only CenterLoss allocation.
    center = torch.nn.Linear(1, 1).to(device)
    optimizer, _ = make_optimizer(c, student, center)
    scheduler = create_scheduler(c, optimizer, iters_per_epoch=1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    bn_count = int(student.bottleneck.num_batches_tracked)
    teacher_bn = int(teacher.bottleneck.num_batches_tracked)
    with torch.no_grad():
        logits_teacher = normalized_clip_logits(
            teacher.forward_pre_bn(joined, modalities, platforms), teacher.base.clip_proj, text,
            c.M2.TEXT_TEMPERATURE)
    score, g, _ = student(images, target, cameras)
    kd = kd_kl(logits_teacher, normalized_clip_logits(g, student.base.clip_proj, text,
        c.M2.TEXT_TEMPERATURE), c.M2.KD_TEMPERATURE)
    gradients = torch.autograd.grad(kd, [student.base.patch_embed.proj.weight,
        student.bottleneck.weight, student.classifier.weight], retain_graph=True, allow_unused=True)
    assert torch.isfinite(gradients[0]).all() and gradients[0].norm() > 0
    assert gradients[1] is None and gradients[2] is None
    loss = (c.MODEL.ID_LOSS_WEIGHT * F.cross_entropy(score, target) +
        c.MODEL.TRIPLET_LOSS_WEIGHT * triplet(g, target)[0] + c.M2.KD_WEIGHT * kd)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.base.clip_proj.grad is None
    assert all(p.grad is None for p in teacher.parameters())
    assert int(student.bottleneck.num_batches_tracked) == bn_count + 1
    assert int(teacher.bottleneck.num_batches_tracked) == teacher_bn
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
    optimizer.step()
    print('BASELINE_KD_STEP_OK: original CE/Triplet + KD; visual gradient; teacher/P0 frozen; one BN update', flush=True)

    attach_metadata(student, c, bank, payload)
    assert student.m5_metadata['student'] is None
    # This is an epoch1/one-update diagnostic, never an epoch60 result.
    with tempfile.TemporaryDirectory(prefix='baseline-m5-check-') as out:
        save_training(out, 'transformer', 1, student, optimizer, scheduler, scaler, 1, 1, 0, 0.)
        assert restore_training(str(Path(out)/'transformer_1.pth'), student,
            optimizer, scheduler, scaler)[:4] == (2, 1, 1, 0)
        off.MODEL.PRETRAIN_CHOICE = 'no'
        off.MODEL.PRETRAIN_PATH = off.M2.CLIP_PATH = off.M2.TEXT_BANK = off.M2.TEACHER_WEIGHT = 'absent'
        deployment = make_model(off, 500, 7, 0).to(device)
        deployment.load_param(str(Path(out)/'transformer_1_student.pth'))
        student.eval()
        deployment.eval()
        with torch.no_grad():
            for neck in ('before', 'after'):
                student.neck_feat = deployment.neck_feat = neck
                torch.testing.assert_close(student(joined, mode=1), deployment(joined, mode=1), rtol=0, atol=0)
        del deployment
        gc.collect()
        off.MODEL.TOKEN_TRAJECTORY = True
        wrong = make_model(off, 500, 7, 0)
        try:
            wrong.load_param(str(Path(out)/'transformer_1_student.pth'))
        except ValueError:
            pass
        else:
            raise AssertionError('baseline checkpoint accepted by C0 student')
    print('BASELINE_M5_RELOAD_OK: full-state restore, teacher-free pre/post export, rejects C0 mismatch', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--artifacts-only', action='store_true')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(4)
    check(load_config(args.config_file, args.opts), args.device, args.artifacts_only)
    print('Check complete. Not 192-image GPU AMP or formal 60-epoch evidence.', flush=True)
