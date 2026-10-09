"""M2-3 stage-A prompt fitting from a training-only visual feature cache."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import torch
import torch.nn.functional as F
from model.make_model import _read_clip_checkpoint
from model.m2.deep_text_control import DeepIdentityText

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', required=True); p.add_argument('--feature-cache', required=True)
    p.add_argument('--output', required=True); p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--ids-per-step', type=int, default=16)
    p.add_argument('--images-per-id', type=int, default=4)
    p.add_argument('--lr', type=float, default=3.5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.07)
    a = p.parse_args(); cache = torch.load(a.feature_cache, map_location='cpu', weights_only=False)
    features = cache.get('features', cache.get('feature'))
    labels = cache.get('labels', cache.get('pid'))
    if features is None or labels is None:
        raise KeyError('feature cache must contain features/labels (or feature/pid)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat = F.normalize(features.float(), dim=-1).to(device)
    labels = torch.as_tensor(labels).long()
    classes = torch.unique(labels, sorted=True)
    expected = torch.arange(classes.numel(), dtype=classes.dtype)
    if not torch.equal(classes.cpu(), expected):
        raise ValueError('training PID labels must be contiguous 0..N-1 for the text bank')
    n_cls = int(classes.numel())
    if n_cls < 2:
        raise ValueError('prompt stage needs at least two training identities')
    mod = DeepIdentityText(_read_clip_checkpoint(a.clip), n_cls,
                           train_id_tokens=True).to(device)
    with torch.no_grad():
        mod.id_bank.normal_(mean=0.0, std=0.02)
    # Stage A trains S1..S4 only.  D1..D4 remain the fixed X embedding and
    # deep contexts are not part of this stage.
    opt = torch.optim.Adam([mod.id_bank], lr=a.lr, weight_decay=a.weight_decay)
    mod.train(); updates = 0
    rows_by_id = [torch.nonzero(labels == i, as_tuple=False).flatten() for i in range(n_cls)]
    # §2.4 samples one real modality×platform condition per update.  The
    # fallback keeps old caches usable but records that they lacked metadata.
    cond = None
    if 'modality' in cache and 'platform' in cache:
        cond = {}
        for i, (m, pform) in enumerate(zip(cache['modality'], cache['platform'])):
            cond.setdefault((int(m), int(pform)), []).append(i)
        cond = {k: torch.tensor(v, dtype=torch.long) for k, v in cond.items()
                if len(torch.unique(labels[v])) >= 2}
    if not cond:
        cond = {('all', 'all'): torch.arange(labels.numel())}
    while updates < a.steps:
        pool = cond[list(cond)[int(torch.randint(len(cond), (1,)))]]
        pool_ids = torch.unique(labels[pool])
        k = min(a.ids_per_step, len(pool_ids))
        if k < 2:
            raise ValueError('a valid condition has fewer than two identities')
        ids = pool_ids[torch.randperm(len(pool_ids))[:k]]
        rows = torch.cat([pool[torch.nonzero(labels[pool] == i, as_tuple=False).flatten()
                          [torch.randint(int((labels[pool] == i).sum()), (a.images_per_id,))]]
                          for i in ids])
        opt.zero_grad()
        text = mod.prompt_forward(ids)
        target = torch.arange(k, device=device).repeat_interleave(a.images_per_id)
        logits = feat[rows.to(device)] @ text.t() / a.temperature
        loss_i2t = F.cross_entropy(logits, target)
        # Multi-positive mean log-probability, §2.4: do not mark other
        # same-identity images as negatives in the text-to-image direction.
        log_t2i = F.log_softmax(logits.t(), dim=1)
        positive = target.unsqueeze(0) == torch.arange(k, device=device).unsqueeze(1)
        loss_t2i = -(log_t2i * positive).sum(1).div(positive.sum(1)).mean()
        loss_id = (loss_i2t + loss_t2i) * 0.5
        loss_id.backward()
        torch.nn.utils.clip_grad_norm_([mod.id_bank], 1.0)
        updates += 1
        if updates <= a.warmup:
            lr = a.lr * updates / max(1, a.warmup)
        else:
            progress = (updates - a.warmup) / max(1, a.steps - a.warmup)
            lr = a.lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        for group in opt.param_groups:
            group['lr'] = lr
        opt.step()
        if updates % 1000 == 0:
            print('step={} loss_id={:.5f} lr={:.7f}'.format(updates, loss_id.item(), lr))
    with torch.no_grad():
        anchors = mod.prompt_forward(torch.arange(n_cls)).detach()
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    torch.save({'id_tokens': mod.id_bank.detach().cpu(), 'anchors': anchors.cpu(),
                'steps': updates, 'template': 'D slots before four ID slots',
                'pid_values': classes.cpu()}, a.output)
    print('saved {}'.format(a.output))

if __name__ == '__main__': main()
