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
    p.add_argument('--lr', type=float, default=3.5e-4); p.add_argument('--anchor-weight', type=float, default=0.1)
    a = p.parse_args(); cache = torch.load(a.feature_cache, map_location='cpu', weights_only=False)
    feat = F.normalize(cache['features'].float(), dim=-1); labels = cache['labels'].long()
    n_cls = int(labels.max().item()) + 1; mod = DeepIdentityText(_read_clip_checkpoint(a.clip), n_cls)
    with torch.no_grad():
        anchor = mod(torch.arange(n_cls)).detach(); mod.text_anchors.copy_(anchor)
    opt = torch.optim.Adam([mod.contexts, mod.id_bank], lr=a.lr); mod.train()
    for step in range(a.steps):
        opt.zero_grad(); ids = torch.unique(labels); text = mod(ids)
        loss_id = F.cross_entropy(feat @ text.t() / 0.07, labels)
        loss_anchor = (1 - (text * F.normalize(anchor[ids], dim=-1)).sum(-1)).mean()
        loss = loss_id + a.anchor_weight * loss_anchor; loss.backward(); opt.step()
        if (step + 1) % 1000 == 0: print('step={} loss_id={:.5f} loss_anchor={:.5f}'.format(step + 1, loss_id.item(), loss_anchor.item()))
    torch.save({'id_tokens': mod.id_bank.detach().cpu(), 'anchors': anchor.cpu(), 'steps': a.steps}, a.output)
    print('saved {}'.format(a.output))

if __name__ == '__main__': main()
