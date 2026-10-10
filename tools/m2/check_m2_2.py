"""M2-2 方法检查：真实权重和缓存；只做小批量检查，不执行正式训练。"""
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
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.m2_losses import ground_loss, xit_loss
from loss.triplet_loss import TripletLoss
from utils.m2_artifacts import train_manifest, PREPROCESS

def check(cfg, device):
    torch.set_num_threads(2)
    device = torch.device(device)
    cfg = cfg.clone()
    cfg.freeze()
    torch.manual_seed(cfg.SOLVER.SEED)
    m = make_model(cfg, num_class=len(train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)[1]),
                   camera_num=7, view_num=3).to(device)
    rng_enabled = torch.get_rng_state()
    saved = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()
             if k.startswith(('base.', 'classifier.', 'bottleneck.'))}
    off = cfg.clone()
    off.defrost()
    off.M2.ENABLED = False
    off.M2.CLIP_PATH = '/missing/teacher'
    off.M2.RGB_CENTERS = '/missing/centers'
    off.freeze()
    torch.manual_seed(cfg.SOLVER.SEED)
    ref = make_model(off, num_class=m.num_classes, camera_num=7, view_num=3).to(device)
    assert torch.equal(rng_enabled, torch.get_rng_state()), 'M2 init consumed main RNG'
    assert all(torch.equal(v, ref.state_dict()[k].cpu()) for k, v in saved.items()), 'original parameter init changed'
    assert ref.m2_inverter is None and ref.m2_text is None
    del saved
    rows, _ = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)
    images = []
    tfm = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                              transforms.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])
    for mod in range(3):
        current = []
        for pid in range(2):
            path = next(r[0] for r in rows if r[1] == pid and r[3] == mod)
            with Image.open(path) as im:
                current.append(tfm(im.convert('RGB')))
        images.append(torch.stack(current).to(device))
    m.eval()
    ref.eval()
    with torch.no_grad():
        for readout in ('before', 'after'):
            m.neck_feat = ref.neck_feat = readout
            assert torch.equal(m(images[0], mode=1), ref(images[0], mode=1)), 'M2 eval added computation'
        _, summary = m.base(torch.cat(images), return_trajectory_summary=True)
        assert summary.shape == (6, 11, 2304)
        assert torch.count_nonzero(summary[:, :, 768:]) == 0, 'zero gain correction summary is nonzero'
    del ref
    gc.collect()
    m.train()
    assert not m.m2_text.training
    labels = torch.tensor([0, 1], device=device)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        score, g, f, aux = m(images, labels)
    summary = aux['summary']
    words, attn = m.m2_inverter(summary)
    assert 'x_embedding' in dict(m.m2_inverter.named_buffers())
    expected = m.m2_inverter.norm(F.gelu(m.m2_inverter.in_proj(summary))) + m.m2_inverter.layer_code[None]
    expected_attn = torch.softmax(torch.einsum('kw,nlw->nkl', m.m2_inverter.query, expected) / 128 ** 0.5, -1)
    torch.testing.assert_close(attn, expected_attn)
    embeddings, eot = m.m2_inverter.splice(words)
    text = F.normalize(m.m2_text(embeddings, torch.full((6,), eot, device=device, dtype=torch.long)).float(), dim=-1)
    z = F.normalize(g.float() @ m.base.clip_proj.float(), dim=-1)
    pids = labels.repeat(3)
    modalities = torch.arange(3, device=device).repeat_interleave(2)
    ground = ground_loss(text, pids, m.m2_rgb_centers)
    xit = xit_loss(z, text, pids, modalities)
    allowed = [m.base.token_trajectory.gain, m.m2_inverter.in_proj.weight,
               m.m2_inverter.word.weight, m.base.patch_embed.proj.weight]
    grads = torch.autograd.grad(ground, allowed, retain_graph=True)
    assert all(torch.isfinite(v).all() and v.abs().sum() > 0 for v in grads), 'ground gradient path broken'
    forbidden = torch.autograd.grad(xit, [m.m2_inverter.word.weight, m.m2_inverter.query],
                                    retain_graph=True, allow_unused=True)
    assert all(v is None for v in forbidden), 'xit updated text target generator'
    assert not m.base.clip_proj.requires_grad
    assert all(not p.requires_grad for p in m.m2_text.parameters())
    # Independently evaluate the exact average-log multi-positive formula.
    positive_terms = []
    for i in range(6):
        candidates = modalities != modalities[i]
        positive = pids[candidates] == pids[i]
        logits = z[i] @ text.detach()[candidates].t() / 0.07
        positive_terms.append(-F.log_softmax(logits, 0)[positive].mean())
    torch.testing.assert_close(xit, torch.stack(positive_terms).mean())
    # Full PKM mask: 128 candidates, 8 positives, 120 other IDs.
    pid192 = torch.arange(16).repeat_interleave(4).repeat(3)
    mod192 = torch.arange(3).repeat_interleave(64)
    candidates = mod192[:, None] != mod192[None]
    positives = candidates & (pid192[:, None] == pid192[None])
    assert (candidates.sum(1) == 128).all() and (positives.sum(1) == 8).all()
    center = torch.nn.Linear(2, 2)
    optimizer, _ = make_optimizer(cfg, m, center)
    groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
    assert groups[id(m.m2_inverter.word.weight)]['lr'] == 3.5e-4
    assert groups[id(m.base.token_trajectory.gain)]['lr'] == 3.5e-4
    assert groups[id(m.base.patch_embed.proj.weight)]['lr'] == 5e-6
    assert id(m.base.clip_proj) not in groups
    assert not any(id(p) in groups for p in m.m2_text.parameters())
    assert len(groups) == sum(len(group['params']) for group in optimizer.param_groups)
    total = F.cross_entropy(score, pids) + TripletLoss()(g, pids)[0] + 0.5 * ground + 0.5 * xit
    total.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    assert all(p.grad is None for p in m.m2_text.parameters())
    assert m.base.clip_proj.grad is None
    # Perturb gain so export/reload isn't merely checking all-zero C0.
    with torch.no_grad():
        m.base.token_trajectory.gain.normal_(std=0.001)
    return m, images[0], optimizer, center

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    config = defaults.clone()
    config.merge_from_file(args.config)
    config.merge_from_list(args.opts)
    model, _, _, _ = check(config, args.device)
    print('M2-2 小批量方法检查通过：配置/形状/零 gain/梯度/冻结/关闭路径/参数 LR。')
    print('本检查没有启动正式训练，也不代表方法取得检索增益。')

if __name__ == '__main__':
    main()
