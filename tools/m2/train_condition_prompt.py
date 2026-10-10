"""M2-1 条件均衡阶段 A：唯一身份文本与双向多正例目标。"""
import argparse
import math
import os
import random
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from config import cfg as defaults
from model.make_model import _read_clip_checkpoint
from model.m2.condition_prompt import ConditionPrompt
from utils.m1_artifacts import validate_cache, file_hash, VERSION, BASE_COMMIT

class ConditionCycle:
    def __init__(self, labels, modalities, platforms, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.pools, self.skipped, self.order = {}, {}, []
        for condition in sorted(set(zip(modalities.tolist(), platforms.tolist()))):
            rows = ((modalities == condition[0]) & (platforms == condition[1])).nonzero(as_tuple=False).flatten()
            ids = labels[rows].unique().tolist()
            if len(ids) < 2:
                self.skipped[condition] = len(ids)
            else:
                self.pools[condition] = {y: rows[labels[rows] == y] for y in ids}
        if not self.pools:
            raise ValueError('no real condition has two train identities')
        self.conditions = sorted(self.pools)
        self.counts = {condition: 0 for condition in self.conditions}

    def sample(self, identities, images_per_id):
        if not self.order:
            self.order = torch.randperm(len(self.conditions), generator=self.generator).tolist()
        condition = self.conditions[self.order.pop()]
        self.counts[condition] += 1
        pool = self.pools[condition]
        ids = torch.tensor(sorted(pool))
        ids = ids[torch.randperm(len(ids), generator=self.generator)[:min(identities, len(ids))]]
        selected = []
        for y in ids.tolist():
            rows = pool[y]
            index = (torch.randperm(len(rows), generator=self.generator)[:images_per_id]
                     if len(rows) >= images_per_id else
                     torch.randint(len(rows), (images_per_id,), generator=self.generator))
            selected.append(rows[index])
        return condition, ids, torch.cat(selected)

def prompt_loss(features, text, images_per_id, temperature):
    with torch.cuda.amp.autocast(enabled=False):
        targets = torch.arange(len(text), device=text.device).repeat_interleave(images_per_id)
        scores = F.normalize(features.float(), dim=1) @ F.normalize(text.float(), dim=1).t() / temperature
        image_to_text = F.cross_entropy(scores, targets)
        positive = targets[None] == torch.arange(len(text), device=text.device)[:, None]
        log_probability = F.log_softmax(scores.t(), dim=1)
        text_to_image = -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
        return .5 * (image_to_text + text_to_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    for arg in ('cache', 'clip', 'output', 'data-root', 'prompts-output'):
        parser.add_argument('--' + arg)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = defaults.clone()
    cfg.merge_from_file(args.config_file)
    if args.steps is not None:
        cfg.PROMPT_STAGE.MAX_UPDATES = args.steps
    if args.warmup is not None:
        cfg.PROMPT_STAGE.WARMUP_UPDATES = args.warmup
    cfg.M2.CLIP_PATH = args.clip or cfg.M2.CLIP_PATH
    cfg.MODEL.PRETRAIN_PATH = cfg.M2.CLIP_PATH
    cfg.M2.FEATURE_CACHE = args.cache or cfg.M2.FEATURE_CACHE
    cfg.M2.TEXT_BANK = args.output or cfg.M2.TEXT_BANK
    cfg.M2.ID_PROMPT_WEIGHT = args.prompts_output or cfg.M2.ID_PROMPT_WEIGHT
    cfg.DATASETS.ROOT_DIR = args.data_root or cfg.DATASETS.ROOT_DIR
    stage = cfg.PROMPT_STAGE
    if not stage.CONDITION_BALANCED or (cfg.M2.ID_CONTEXT_LENGTH, cfg.M2.MODALITY_CONTEXT_LENGTH, cfg.M2.PLATFORM_CONTEXT_LENGTH) != (4, 2, 2):
        raise ValueError('M2-1 requires condition-balanced sampling and 4/2/2 tokens')
    seed = cfg.SOLVER.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)
    cache = torch.load(cfg.M2.FEATURE_CACHE, map_location='cpu', weights_only=False)
    mapping = validate_cache(cache, cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR, cfg.M2.CLIP_PATH)
    sampler = ConditionCycle(cache['pid'], cache['modality'], cache['platform'], seed)
    print('阶段 A 有效条件:', sampler.conditions, '跳过条件:', sampler.skipped, flush=True)
    model = ConditionPrompt(_read_clip_checkpoint(cfg.M2.CLIP_PATH), len(mapping)).to(device).train()
    parameters = [model.identity, model.modality, model.platform]
    optimizer = torch.optim.Adam(parameters, lr=stage.LEARNING_RATE, weight_decay=stage.WEIGHT_DECAY)
    features = cache['feature'].float().to(device)
    updates = 0
    while updates < stage.MAX_UPDATES:
        condition, ids, rows = sampler.sample(stage.IDENTITIES_PER_STEP, stage.IMAGES_PER_ID)
        step = updates + 1
        if stage.WARMUP_UPDATES and step <= stage.WARMUP_UPDATES:
            lr = stage.LEARNING_RATE * step / stage.WARMUP_UPDATES
        else:
            progress = (step - stage.WARMUP_UPDATES) / max(1, stage.MAX_UPDATES - stage.WARMUP_UPDATES)
            lr = stage.LEARNING_RATE * .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad()
        ids = ids.to(device)
        text = model(ids, torch.full_like(ids, condition[0]), torch.full_like(ids, condition[1]))
        loss = prompt_loss(features[rows.to(device)], text, stage.IMAGES_PER_ID, stage.TEMPERATURE)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite prompt loss; no final artifact exported')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        updates += 1
        if updates % 1000 == 0 or updates == stage.MAX_UPDATES:
            print('阶段 A update={} loss={:.6f} lr={:.8f}'.format(updates, loss.item(), lr), flush=True)
    conditions = cache['metadata']['condition_mask'].nonzero(as_tuple=False)
    model.eval()
    with torch.no_grad():
        prototypes = torch.cat([model(*batch.to(device).unbind(1)).cpu() for batch in conditions.split(32)])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[2]), text=True).strip()
    metadata = {'variant': 'condition_text_bridge', 'implementation': VERSION, 'base_commit': BASE_COMMIT,
                'code_commit': commit, 'config': cfg.dump(), 'signature': model.signature,
                'metadata': cache['metadata'], 'feature_cache_sha256': file_hash(cfg.M2.FEATURE_CACHE),
                'stage_config': dict(stage), 'seed': seed, 'updates': updates,
                'actual_iterations': updates, 'optimizer_updates': updates, 'amp_skips': 0,
                'condition_updates': sampler.counts, 'skipped_conditions': sampler.skipped}
    bank = dict(metadata, text=prototypes, pid=conditions[:, 0], modality=conditions[:, 1], platform=conditions[:, 2])
    prompts = dict(metadata, identity=model.identity.detach().cpu(), modality=model.modality.detach().cpu(),
                   platform=model.platform.detach().cpu(), optimizer=optimizer.state_dict())
    for path, artifact in ((cfg.M2.ID_PROMPT_WEIGHT, prompts), (cfg.M2.TEXT_BANK, bank)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(artifact, path)
    print('已导出最终条件 prompt 和真实条件文本银行；实际更新数:', updates, flush=True)

if __name__ == '__main__':
    main()
