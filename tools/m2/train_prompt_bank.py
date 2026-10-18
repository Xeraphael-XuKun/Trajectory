"""M2-5 ordinary P-ID: balanced observed conditions, real optimizer steps."""
import argparse
import math
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from model.backbones.clip_text import CLIPTextEncoder
from tools.m2.artifacts import (VERSION, BASE_COMMIT, TEMPLATE, read_clip, file_digest, rng_state,
    restore_rng, seed_stage, code_commit, load_config, validate_cache, template_tokens)

def prompt_loss(z, text, row_pids, text_pids, temperature=.07):
    with torch.cuda.amp.autocast(enabled=False):
        logits = z.float() @ text.float().t() / temperature
        target = (row_pids[:, None] == text_pids[None, :]).long().argmax(1)
        image_loss = F.cross_entropy(logits, target)
        log_prob = logits.t().log_softmax(dim=1)
        positives = text_pids[:, None] == row_pids[None, :]
        text_loss = -((log_prob * positives).sum(1) / positives.sum(1)).mean()
        return .5 * (image_loss + text_loss)

def train(cfg, device='cuda'):
    seed_stage(cfg)
    cache = torch.load(cfg.M2.FEATURE_CACHE, map_location='cpu', weights_only=False)
    source = validate_cache(cache, cfg)
    z, pids, mods, plats = (cache[k] for k in ('clip_features', 'pids', 'modalities', 'platforms'))
    y = len(source['pid_map'])
    groups, skipped = {}, []
    for condition in sorted(set(zip(mods.tolist(), plats.tolist()))):
        mask = (mods == condition[0]) & (plats == condition[1])
        members = {int(pid): torch.where(mask & (pids == pid))[0] for pid in pids[mask].unique()}
        if len(members) < 2:
            skipped.append(condition)
        else:
            groups[condition] = members
    if not groups:
        raise ValueError('no condition contains two train identities')
    te = CLIPTextEncoder().to(device)
    te.load_clip(read_clip(cfg.M2.CLIP_PATH))
    te.eval()
    ids, slots, eot = template_tokens(device)
    context = torch.nn.Parameter(torch.randn(y, 4, 512, device=device) * .02)
    opt = torch.optim.Adam([context], lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    steps = cfg.M2.PROMPT_STEPS
    schedule = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda step: min(1., (step + 1) / 100.) if step < 100 else
        .5 * (1 + math.cos(math.pi * (step - 100) / max(1, steps - 100))))
    meta = {'implementation': VERSION, 'stage': 'prompt', 'base_commit': BASE_COMMIT,
        'code_commit': code_commit(), 'source': source, 'config': cfg.dump(),
        'template': TEMPLATE, 'token_slots': slots, 'eot_index': eot,
        'temperature': cfg.M2.TEXT_TEMPERATURE, 'cache_sha256': file_digest(cfg.M2.FEATURE_CACHE)}
    counts, conditions, begin, elapsed = {str(k): 0 for k in groups}, [], 0, 0.
    if cfg.M2.PROMPT_RESUME:
        previous = torch.load(cfg.M2.PROMPT_RESUME, map_location=device, weights_only=False)
        from utils.m5_checkpoint import compare_config
        compare_config(previous['metadata']['config'], cfg.dump())
        if previous['metadata']['cache_sha256'] != meta['cache_sha256']:
            raise ValueError('prompt resume source cache differs')
        context.data.copy_(previous['tokens'])
        opt.load_state_dict(previous['optimizer'])
        schedule.load_state_dict(previous['scheduler'])
        begin = previous['metadata']['optimizer_updates']
        counts, conditions = previous['budget']['condition_steps'], previous['conditions_remaining']
        elapsed = previous['budget']['seconds']
        state = previous['rng']
        state['torch'] = state['torch'].cpu()
        state['cuda'] = [s.cpu() for s in state['cuda']]
        restore_rng(state)

    def encode(rows):
        embeddings = te.token_embedding(ids[None].expand(len(rows), -1)).clone()
        embeddings[:, slots] = context[rows]
        return F.normalize(te(embeddings, torch.full((len(rows),), eot, device=device, dtype=torch.long)).float(), dim=1)

    def save(step, seconds, path):
        payload = {'tokens': context.detach().cpu(), 'metadata': dict(meta, optimizer_updates=step),
            'optimizer': opt.state_dict(), 'scheduler': schedule.state_dict(), 'rng': rng_state(),
            'conditions_remaining': conditions.copy(), 'budget': {'seconds': seconds,
                'optimizer_steps': step, 'condition_steps': counts.copy(), 'skipped_conditions': skipped}}
        torch.save(payload, path)
        return payload

    Path(cfg.M2.PROMPT_WEIGHT).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for step in range(begin, steps):
        if not conditions:
            conditions = list(groups)
            random.shuffle(conditions)
        condition = conditions.pop()
        members = groups[condition]
        selected = random.sample(sorted(members), min(16, len(members)))
        indices = []
        for pid in selected:
            available = members[pid].tolist()
            indices.extend(random.sample(available, 4) if len(available) >= 4 else random.choices(available, k=4))
        rows = torch.tensor(selected, device=device)
        loss = prompt_loss(z[indices].to(device), encode(rows), pids[indices].to(device), rows, cfg.M2.TEXT_TEMPERATURE)
        opt.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([context], 1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise ValueError('nonfinite prompt update; budget not advanced')
        opt.step()
        schedule.step()
        counts[str(condition)] += 1
        if (step + 1) % cfg.SOLVER.LOG_PERIOD == 0:
            print('prompt_update=%d loss=%.6f condition=%s' % (step + 1, loss.item(), condition), flush=True)
        if (step + 1) % cfg.M2.PROMPT_SAVE_PERIOD == 0:
            save(step + 1, elapsed + time.time() - start, str(Path(cfg.M2.PROMPT_WEIGHT).with_name('prompt_resume.pt')))
    payload = save(steps, elapsed + time.time() - start, cfg.M2.PROMPT_WEIGHT)
    with torch.no_grad():
        text = torch.cat([encode(torch.arange(i, min(i + 32, y), device=device)) for i in range(0, y, 32)]).cpu()
    payload['text_bank'] = text
    torch.save(payload, cfg.M2.PROMPT_WEIGHT)
    bank = {'text_bank': text, 'metadata': dict(payload['metadata'], prompt_sha256=file_digest(cfg.M2.PROMPT_WEIGHT))}
    torch.save(bank, cfg.M2.TEXT_BANK)
    print('final prompt updates=%d text=%s skipped=%s' % (steps, cfg.M2.TEXT_BANK, skipped), flush=True)
    return bank

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('opts', nargs=argparse.REMAINDER)
    a = p.parse_args()
    train(load_config(a.config_file, a.opts), a.device)

if __name__ == '__main__':
    main()
