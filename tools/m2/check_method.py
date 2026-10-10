"""Numerical/gradient and real-CLIP M2-5 checks; never formal training."""
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
from model.m2_teacher import ConditionalQVLora, ConditionalVisualTeacher, normalized_clip_logits, kd_kl
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from tools.m2.artifacts import load_config, train_source, validate_text_bank, seed_stage
from tools.m2.train_prompt_bank import prompt_loss
from utils.m5_checkpoint import attach_metadata, save_training, restore_training, load_teacher

def formulas(device):
    torch.manual_seed(7)
    lora = ConditionalQVLora(dim=8, depth=2, rank=3).to(device)
    for name, param in lora.named_parameters():
        if name.endswith('B'):
            param.data.normal_(0, .1)
    x = torch.randn(6, 5, 8, device=device, requires_grad=True)
    mods, plats = torch.tensor([2, 0, 1, 2, 0, 1], device=device), torch.tensor([1, 0, 1, 0, 1, 0], device=device)
    actual = lora.qkv_delta(1, mods, plats, x)
    expected = []
    for i in range(len(x)):
        q, v = lora.delta(1, int(mods[i]), int(plats[i]))
        expected.append(torch.cat([F.linear(x[i], q), torch.zeros_like(x[i]), F.linear(x[i], v)], -1))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected)
    params = [x] + list(lora.parameters())
    a = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), params)
    for one, two in zip(a, b):
        torch.testing.assert_close(one, two, rtol=1e-4, atol=1e-6)
    assert torch.count_nonzero(actual[:, :, 8:16]) == 0
    teacher = torch.randn(9, 5, device=device, requires_grad=True)
    student = torch.randn(9, 5, device=device, requires_grad=True)
    td = 2.3
    reference = td ** 2 * (F.softmax(teacher.detach()/td, -1) *
        (F.log_softmax(teacher.detach()/td, -1) - F.log_softmax(student/td, -1))).sum(-1).mean()
    actual = kd_kl(teacher, student, td)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, student, retain_graph=True)[0], torch.autograd.grad(reference, student)[0])
    assert torch.autograd.grad(actual, teacher, allow_unused=True)[0] is None
    z, text = F.normalize(torch.randn(6, 12, device=device), dim=1), F.normalize(torch.randn(3, 12, device=device), dim=1)
    pids, labels = torch.tensor([9, 2, 9, 7, 2, 7], device=device), torch.tensor([7, 9, 2], device=device)
    logits = z @ text.t()/.07
    i2t = sum(-F.log_softmax(logits, 1)[i, int(torch.where(labels == pids[i])[0])] for i in range(6))/6
    t2i = sum(-F.log_softmax(logits.t(), 1)[j, pids == labels[j]].mean() for j in range(3))/3
    torch.testing.assert_close(prompt_loss(z, text, pids, labels), .5*(i2t+t2i))
    print('LoRA Q/V matrix convention, both factor/input gradients, K untouched, KD Td^2 and prompt multi-positive formulas PASS', flush=True)

def check(cfg, device='cpu', export_dir=None):
    formulas(device)
    seed_stage(cfg)
    bank = torch.load(cfg.M2.TEXT_BANK, map_location='cpu', weights_only=False)
    source = validate_text_bank(bank, cfg)
    classes = len(source['pid_map'])
    rows, _ = train_source(cfg)
    ids = sorted({row[1] for row in rows})[:3]
    if len(ids) < 3:
        raise ValueError('method check needs three actual train identities')
    tf = transforms.Compose([transforms.Resize(cfg.INPUT.SIZE_TEST), transforms.ToTensor(),
        transforms.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])
    images = []
    cameras = []
    for modality in range(3):
        group, cams = [], []
        for y in ids:
            row = next(r for r in rows if r[1] == y and r[3] == modality)
            with Image.open(row[0]) as image:
                group.append(tf(image.convert('RGB')))
            cams.append(row[2])
        images.append(torch.stack(group).to(device))
        cameras.append(torch.tensor(cams, device=device))
    target = torch.tensor(ids, device=device).repeat(3)
    text = bank['text_bank'].to(device).detach()
    joined = torch.cat(images)
    mods = torch.arange(3, device=device).repeat_interleave(3)
    cams = torch.cat(cameras)
    plats = ((cams == 5) | (cams == 6)).long()
    # These temporary one-epoch/two-update checkpoints are diagnostics, not epoch60 artifacts.
    c = cfg.clone()
    c.defrost()
    c.SOLVER.MAX_EPOCHS = c.TEACHER_STAGE.MAX_EPOCHS = 1
    student = make_model(c, classes, 7, 0).to(device)
    rng = torch.get_rng_state().clone()
    seed_stage(c)
    off = c.clone()
    off.M2.ENABLED = False
    baseline = make_model(off, classes, 7, 0).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
    del baseline
    gc.collect()
    assert not student.base.clip_proj.requires_grad
    seed_stage(c)
    teacher = ConditionalVisualTeacher(c, classes).to(device)
    assert teacher.base.token_trajectory is None
    assert sum(p.numel() for p in teacher.lora.parameters()) == 1474560
    assert not teacher.bottleneck.bias.requires_grad and teacher.classifier.bias is None
    student.eval(); teacher.eval()
    with torch.no_grad():
        zero = teacher.forward_pre_bn(joined, mods, plats)
        torch.testing.assert_close(zero, teacher.base(joined), rtol=0, atol=0)
        torch.testing.assert_close(zero, student.base(joined), rtol=0, atol=0)
    frozen = {key: p.detach().cpu().clone() for key, p in teacher.base.named_parameters()}
    criterion = TripletLoss()
    dummy = torch.nn.Linear(1, 1)
    opt, _ = make_optimizer(c, teacher, dummy)
    assert all(group['lr'] == c.SOLVER.BASE_LR for group in opt.param_groups)
    scheduler = create_scheduler(c, opt, iters_per_epoch=2)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    teacher.train()
    assert teacher.base.training and any(getattr(b.drop_path, 'drop_prob', 0) > 0 for b in teacher.base.blocks)
    for step in range(2):
        scheduler.step_update(step)
        opt.zero_grad()
        score, g, f = teacher(images, camids=cameras)
        auxiliary = F.cross_entropy(normalized_clip_logits(g, teacher.base.clip_proj, text), target)
        named = list(teacher.named_parameters())
        trainable = [(name, p) for name, p in named if p.requires_grad]
        gradients = torch.autograd.grad(auxiliary, [p for _, p in trainable], retain_graph=True, allow_unused=True)
        grad = dict((name, value) for (name, _), value in zip(trainable, gradients))
        assert grad['bottleneck.weight'] is None and grad['classifier.weight'] is None
        assert sum(float(value.norm()) for name, value in grad.items() if name.startswith('lora.') and name.endswith('B')) > 0
        asum = sum(float(value.norm()) for name, value in grad.items() if name.startswith('lora.') and name.endswith('A'))
        assert asum == 0 if step == 0 else asum > 0
        loss = F.cross_entropy(score, target) + criterion(g, target)[0] + .5*auxiliary
        loss.backward()
        assert teacher.bottleneck.weight.grad.norm() > 0 and teacher.classifier.weight.grad.norm() > 0
        assert all(p.grad is None for p in teacher.base.parameters())
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.)
        opt.step()
    for key, p in teacher.base.named_parameters():
        torch.testing.assert_close(p.cpu(), frozen[key], rtol=0, atol=0)
    print('Real CLIP zero LoRA/C0 equivalence, original heads, train DropPath, B then A gradients, frozen visual/P0 and teacher optimizer PASS', flush=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
        p.grad = None
    student.train()
    bn_start = int(student.bottleneck.num_batches_tracked)
    opt_s, _ = make_optimizer(c, student, dummy)
    assert len({id(p) for group in opt_s.param_groups for p in group['params']}) == len(opt_s.param_groups)
    for name, p in student.named_parameters():
        if p.requires_grad:
            group = next(g for g in opt_s.param_groups if g['params'][0] is p)
            assert group['lr'] == (c.SOLVER.PRETRAINED_LR if name.startswith('base.') and '.token_trajectory.' not in name else c.SOLVER.BASE_LR)
    scheduler_s = create_scheduler(c, opt_s, iters_per_epoch=1)
    score, g, f = student(images, camids=cameras)
    with torch.no_grad():
        lt = normalized_clip_logits(teacher.forward_pre_bn(joined, mods, plats), teacher.base.clip_proj, text)
    ls = normalized_clip_logits(g, student.base.clip_proj, text)
    kd = kd_kl(lt, ls, c.M2.KD_TEMPERATURE)
    subset = [student.base.patch_embed.proj.weight, student.base.token_trajectory.gain,
        student.bottleneck.weight, student.classifier.weight]
    grads = torch.autograd.grad(kd, subset, retain_graph=True, allow_unused=True)
    assert all(torch.isfinite(grad).all() and grad.norm() > 0 for grad in grads[:2])
    assert grads[2] is None and grads[3] is None
    (F.cross_entropy(score, target) + criterion(g, target)[0] + c.M2.KD_WEIGHT*kd).backward()
    assert int(student.bottleneck.num_batches_tracked) == bn_start + 1
    assert all(p.grad is None for p in teacher.parameters())
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
    opt_s.step()
    assert student.base.token_trajectory.gain.norm() > 0
    print('Student pre-BN KD reaches visual/gain, excludes BN/classifier/P0/teacher, one BN and one combined update PASS', flush=True)
    if export_dir:
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        c.M2.TRAIN_STAGE = 'teacher'
        c.MODEL.TOKEN_TRAJECTORY = False
        attach_metadata(teacher, c, bank)
        saved = save_training(str(out), 'transformer', 1, teacher, opt, scheduler, scaler, 2, 2, 0, 0.)
        original_bn = teacher.bottleneck.weight.detach().clone()
        teacher.bottleneck.weight.data.add_(1)
        restored = restore_training(str(out/'conditional_teacher_epoch1.pth'), teacher, opt, scheduler, scaler)
        assert restored[:4] == (2, 2, 2, 0)
        torch.testing.assert_close(teacher.bottleneck.weight, original_bn)
        c.M2.TRAIN_STAGE = 'student'; c.MODEL.TOKEN_TRAJECTORY = True
        c.M2.TEACHER_WEIGHT = str(out/'conditional_teacher_epoch1.pth')
        loaded, obj = load_teacher(c, classes, device)
        with torch.no_grad():
            torch.testing.assert_close(teacher.forward_pre_bn(joined, mods, plats), loaded.forward_pre_bn(joined, mods, plats), rtol=0, atol=0)
        del loaded
        attach_metadata(student, c, bank, obj)
        save_training(str(out), 'transformer', 1, student, opt_s, scheduler_s, scaler, 1, 1, 0, 0.)
        assert restore_training(str(out/'transformer_1.pth'), student, opt_s, scheduler_s, scaler)[:4] == (2, 1, 1, 0)
        student.eval()
        off = c.clone(); off.M2.ENABLED = False; off.MODEL.PRETRAIN_CHOICE = 'no'
        off.MODEL.PRETRAIN_PATH = off.M2.CLIP_PATH = off.M2.TEXT_BANK = off.M2.TEACHER_WEIGHT = 'absent-artifact'
        deployment = make_model(off, classes, 7, 0).to(device)
        deployment.load_param(str(out/'transformer_1_student.pth'))
        deployment.eval()
        with torch.no_grad():
            for neck in ('before', 'after'):
                student.neck_feat = deployment.neck_feat = neck
                torch.testing.assert_close(student(joined, mode=1), deployment(joined, mode=1), rtol=0, atol=0)
        deployment.trajectory_accel_mix = 1.
        try:
            deployment.load_param(str(out/'transformer_1_student.pth'))
        except ValueError:
            pass
        else:
            raise AssertionError('accepted wrong C0 acceleration')
        print('Teacher/student full-state restore, strict frozen teacher load, teacher-free pre/post-BN export and scalar mismatch rejection PASS', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('opts', nargs=argparse.REMAINDER)
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='m2-5-method-') as out:
        check(load_config(a.config_file, a.opts), a.device, out)
    print('Method check complete. Not full192/GPUAMP/10000prompt/60epoch evidence.', flush=True)

if __name__ == '__main__':
    main()
