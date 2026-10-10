"""M2-1 必要方法检查：集合公式、缺失条件、梯度与学生部署。"""
import argparse
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from config import cfg as defaults
from model import make_model
from model.make_model import _read_clip_checkpoint
from model.m2.condition_prompt import ConditionPrompt
from model.m2.condition_bridge import condition_bridge_loss, load_text_bank
from tools.m2.train_condition_prompt import prompt_loss, ConditionCycle
from utils.m1_artifacts import train_manifest, PREPROCESS
from utils.m1_checkpoint import attach_metadata, save_training, restore_training
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.triplet_loss import TripletLoss

def check_formulas(device):
    # Ragged observed conditions: missing other-spectrum targets and unequal prototype counts.
    bank = {'text': F.normalize(torch.randn(6, 512, device=device), dim=1),
            'pid': torch.tensor([0, 0, 1, 1, 1, 2], device=device),
            'modality': torch.tensor([0, 0, 0, 1, 2, 0], device=device),
            'platform': torch.tensor([0, 1, 0, 0, 1, 0], device=device)}
    z = torch.randn(6, 512, device=device, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
    modalities = torch.tensor([0, 0, 0, 1, 1, 2], device=device)
    loss, stats = condition_bridge_loss(z, labels, modalities, bank, .09, .13)
    sim = F.normalize(z, dim=1) @ bank['text'].t()
    all_scores = torch.stack([torch.logsumexp(sim[:, bank['pid'] == y] / .09, 1)
                             - torch.tensor(float((bank['pid'] == y).sum()), device=device).log() for y in range(3)], 1)
    expected_all = F.cross_entropy(all_scores, labels)
    terms = []
    for i in range(len(z)):
        keep = bank['modality'] != modalities[i]
        candidates = bank['pid'][keep].unique()
        if len(candidates) < 2 or labels[i] not in candidates:
            continue
        scores = torch.stack([torch.logsumexp(sim[i, keep & (bank['pid'] == y)] / .13, 0)
                              - torch.tensor(float((keep & (bank['pid'] == y)).sum()), device=device).log() for y in candidates])
        terms.append(-F.log_softmax(scores, 0)[(candidates == labels[i]).nonzero().item()])
    expected_cross = torch.stack(terms).mean()
    torch.testing.assert_close(stats['all'], expected_all.detach())
    torch.testing.assert_close(stats['cross'], expected_cross.detach())
    assert stats['valid'] == len(terms)
    torch.testing.assert_close(loss, .5 * (expected_all + expected_cross))
    _, zero_stats = condition_bridge_loss(z[:3], labels[:3], modalities[:3], bank)
    assert zero_stats['valid'] == 0 and zero_stats['cross'] == 0
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    # Independent average log probability over all positive images.
    image, text = torch.randn(8, 512, device=device), torch.randn(2, 512, device=device)
    scores = F.normalize(image, dim=1) @ F.normalize(text, dim=1).t() / .07
    target = torch.arange(2, device=device).repeat_interleave(4)
    expected = .5 * (F.cross_entropy(scores, target) -
                    torch.stack([F.log_softmax(scores.t(), 1)[y, target == y].mean() for y in range(2)]).mean())
    torch.testing.assert_close(prompt_loss(image, text, 4, .07), expected)
    cycle = ConditionCycle(torch.tensor([0, 1, 0, 1, 2]), torch.tensor([0, 0, 1, 1, 2]), torch.zeros(5, dtype=torch.long), 1234)
    for _ in range(6):
        _, ids, _ = cycle.sample(16, 4)
        assert len(ids) == len(ids.unique())
    assert set(cycle.counts.values()) == {3} and cycle.skipped == {(2, 0): 1}
    print('双向 prompt 与 all/cross 独立公式、两种温度、缺失目标零项、条件循环 PASS', flush=True)

def check(cfg, device='cpu', export_dir=None):
    torch.set_num_threads(2)
    device = torch.device(device)
    check_formulas(device)
    rows, mapping = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)
    bank = load_text_bank(cfg.M2.TEXT_BANK, cfg, len(mapping), device)
    # A tiny actual text-input backward; do not perform any formal prompt update.
    prompt = ConditionPrompt(_read_clip_checkpoint(cfg.M2.CLIP_PATH), len(mapping)).to(device).train()
    assert sum(p.numel() for p in (prompt.identity, prompt.modality, prompt.platform)) == len(mapping) * 2048 + 5120
    condition = next((m, v) for m in range(3) for v in range(2) if bank['metadata']['condition_mask'][:, m, v].sum() >= 2)
    ids = bank['metadata']['condition_mask'][:, condition[0], condition[1]].nonzero().flatten()[:2].to(device)
    text = prompt(ids, torch.full_like(ids, condition[0]), torch.full_like(ids, condition[1]))
    assert text.shape == (2, 512) and text.dtype == torch.float32
    (text * torch.randn_like(text)).sum().backward()
    for parameter in (prompt.identity, prompt.modality, prompt.platform):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in prompt.text.parameters())
    assert prompt.signature['eot'] == 15 and len(prompt.signature['token_ids']) == 77
    print('真实文本塔 77 token/真实 EOT、S/U/Q 梯度及冻结教师 PASS', flush=True)
    del prompt
    gc.collect()
    torch.manual_seed(cfg.SOLVER.SEED)
    model = make_model(cfg, len(mapping), 7, 3).to(device)
    rng = torch.get_rng_state()
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    off = cfg.clone()
    off.M2.ENABLED = False
    off.M2.TEXT_BANK = off.M2.CLIP_PATH = '/missing/artifact'
    torch.manual_seed(cfg.SOLVER.SEED)
    reference = make_model(off, len(mapping), 7, 3).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, reference.state_dict()[key].cpu()) for key, value in original.items())
    assert all('text' not in key for key in model.state_dict())
    del original
    transform = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                                    transforms.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])
    images = []
    # Two IDs appearing in all three spectra; labels remain the actual train labels.
    sample_ids = [y for y in range(len(mapping)) if all(any(r[1] == y and r[3] == m for r in rows) for m in range(3))][:2]
    assert len(sample_ids) == 2
    for m in range(3):
        current = []
        for y in sample_ids:
            path = next(r[0] for r in rows if r[1] == y and r[3] == m)
            with Image.open(path) as image:
                current.append(transform(image.convert('RGB')))
        images.append(torch.stack(current).to(device))
    model.eval()
    reference.eval()
    with torch.no_grad():
        for readout in ('before', 'after'):
            model.neck_feat = reference.neck_feat = readout
            torch.testing.assert_close(model(images[0], mode=1), reference(images[0], mode=1), rtol=0, atol=0)
    del reference
    gc.collect()
    model.train()
    labels = torch.tensor(sample_ids, device=device)
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        score, g, _ = model(images, labels)
        with torch.cuda.amp.autocast(enabled=False):
            extra, _ = condition_bridge_loss(g.float() @ model.base.clip_proj.float(), labels.repeat(3),
                torch.arange(3, device=device).repeat_interleave(2), bank,
                cfg.M2.TEMPERATURE, cfg.M2.CROSS_TEMPERATURE, cfg.M2.TEXT_ALL_WEIGHT, cfg.M2.TEXT_CROSS_WEIGHT)
    gain_grad = torch.autograd.grad(extra, model.base.token_trajectory.gain, retain_graph=True)[0]
    assert torch.isfinite(gain_grad).all() and gain_grad.abs().sum() > 0
    assert all(value is None for value in torch.autograd.grad(extra,
               [model.bottleneck.weight, model.classifier.weight], retain_graph=True, allow_unused=True))
    optimizer, _ = make_optimizer(cfg, model, torch.nn.Linear(2, 2))
    groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
    assert id(model.base.clip_proj) not in groups
    assert groups[id(model.base.token_trajectory.gain)]['lr'] == 3.5e-4
    assert groups[id(model.base.patch_embed.proj.weight)]['lr'] == 5e-6
    assert len(groups) == sum(len(group['params']) for group in optimizer.param_groups)
    total = F.cross_entropy(score.float(), labels.repeat(3)) + TripletLoss()(g, labels.repeat(3))[0] + extra
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    assert not model.base.clip_proj.requires_grad and model.base.clip_proj.grad is None
    assert model.base.token_trajectory.gain.abs().sum() > 0
    print('主学生初始化/RNG/关闭路径、gain 梯度、无辅助 BN/classifier 梯度、冻结 P0 和 LR PASS', flush=True)
    if export_dir:
        attach_metadata(model, cfg, bank)
        scheduler = create_scheduler(cfg, optimizer, iters_per_epoch=1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        save_training(export_dir, 'check', 1, model, optimizer, scheduler, scaler, 1, 1, 0)
        model.eval()
        features = {}
        with torch.no_grad():
            for readout in ('before', 'after'):
                model.neck_feat = readout
                features[readout] = model(images[0], mode=1).clone()
        assert restore_training(str(Path(export_dir) / 'check_1.pth'), model, optimizer, scheduler, scaler) == (2, 1, 1, 0)
        off.MODEL.PRETRAIN_CHOICE = 'no'
        student = make_model(off, len(mapping), 7, 3).to(device)
        student.load_param(str(Path(export_dir) / 'check_1_student.pth'))
        student.eval()
        with torch.no_grad():
            for readout in ('before', 'after'):
                student.neck_feat = readout
                torch.testing.assert_close(student(images[0], mode=1), features[readout], rtol=0, atol=0)
        student.base.token_trajectory.acceleration_mix = 1.
        try:
            student.load_param(str(Path(export_dir) / 'check_1_student.pth'))
        except ValueError:
            pass
        else:
            raise AssertionError('accepted wrong acceleration mix')
        print('非零 gain 下无文本工件部署 pre/post-BN 完全一致、完整状态恢复、错误系数拒绝 PASS', flush=True)
    print('M2-1 方法检查完成；不是正式 prompt/60epoch 或检索收益验证。', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--export-dir')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = defaults.clone()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    check(cfg, args.device, args.export_dir)

if __name__ == '__main__':
    main()
