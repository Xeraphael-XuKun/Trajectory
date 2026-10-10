"""M2-3 正式阶段 A：条件循环均衡采样，仅训练四个身份 token。"""
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
from model.m2.deep_text_control import DeepIdentityText
from utils.m3_artifacts import validate_cache, file_hash

class ConditionCycle:
    def __init__(self, labels, modalities, platforms, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.pools, self.skipped, self.order = {}, {}, []
        conditions = sorted(set(zip(modalities.tolist(), platforms.tolist())))
        for condition in conditions:
            rows = torch.nonzero((modalities == condition[0]) & (platforms == condition[1]), as_tuple=False).flatten()
            ids = torch.unique(labels[rows]).tolist()
            if len(ids) < 2:
                self.skipped[condition] = len(ids)
                continue
            self.pools[condition] = {pid: rows[labels[rows] == pid] for pid in ids}
        if not self.pools:
            raise ValueError('no real condition has at least two train identities')
        self.conditions = sorted(self.pools)
        self.counts = {condition: 0 for condition in self.conditions}

    def sample(self, ids_per_step, images_per_id):
        if not self.order:
            self.order = torch.randperm(len(self.conditions), generator=self.generator).tolist()
        condition = self.conditions[self.order.pop()]
        self.counts[condition] += 1
        pool = self.pools[condition]
        ids = torch.tensor(sorted(pool))
        ids = ids[torch.randperm(len(ids), generator=self.generator)[:min(ids_per_step, len(ids))]]
        sampled = []
        for pid in ids.tolist():
            rows = pool[pid]
            if len(rows) >= images_per_id:
                index = torch.randperm(len(rows), generator=self.generator)[:images_per_id]
            else:
                index = torch.randint(len(rows), (images_per_id,), generator=self.generator)
            sampled.append(rows[index])
        return condition, ids, torch.cat(sampled)

def prompt_loss(features, text, images_per_id, temperature):
    with torch.cuda.amp.autocast(enabled=False):
        target = torch.arange(text.shape[0], device=text.device).repeat_interleave(images_per_id)
        logits = F.normalize(features.float(), dim=-1) @ F.normalize(text.float(), dim=-1).t() / temperature
        image_to_text = F.cross_entropy(logits, target)
        log_probability = F.log_softmax(logits.t(), dim=1)
        positive = target[None] == torch.arange(text.shape[0], device=text.device)[:, None]
        text_to_image = -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
        return 0.5 * (image_to_text + text_to_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='')
    parser.add_argument('--clip')
    parser.add_argument('--data-root')
    parser.add_argument('--feature-cache')
    parser.add_argument('--output')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--ids-per-step', type=int)
    parser.add_argument('--images-per-id', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--export-only', action='store_true',
                        help='仅从已完成的阶段 A token 文件重新导出，不再执行 optimizer step')
    args = parser.parse_args()
    cfg = defaults.clone()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    for arg, key in [('steps', 'MAX_UPDATES'), ('ids_per_step', 'IDENTITIES_PER_STEP'),
                     ('images_per_id', 'IMAGES_PER_ID'), ('lr', 'LEARNING_RATE'),
                     ('weight_decay', 'WEIGHT_DECAY'), ('warmup', 'WARMUP_UPDATES'),
                     ('temperature', 'TEMPERATURE')]:
        value = getattr(args, arg)
        if value is not None:
            setattr(cfg.PROMPT_STAGE, key, value)
    if not cfg.PROMPT_STAGE.CONDITION_BALANCED:
        raise ValueError('M2-3 requires formal condition-balanced prompt sampling')
    clip = args.clip or cfg.M2.CLIP_PATH
    cache_path = args.feature_cache or cfg.M2.FEATURE_CACHE
    output = args.output or cfg.M2.TEXT_BANK
    root = args.data_root or cfg.DATASETS.ROOT_DIR
    seed = cfg.SOLVER.SEED if args.seed is None else args.seed
    cfg.MODEL.PRETRAIN_PATH = clip
    cfg.M2.CLIP_PATH = clip
    cfg.M2.FEATURE_CACHE = cache_path
    cfg.M2.TEXT_BANK = cfg.M2.ID_PROMPT_WEIGHT = cfg.M2.DEEP_CONTEXT_INIT = output
    cfg.DATASETS.ROOT_DIR = root
    cfg.SOLVER.SEED = seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    pid_map = validate_cache(cache, root, cfg.DATASETS.SUBDIR, clip)
    sampler = ConditionCycle(cache['labels'], cache['modality'], cache['platform'], seed)
    print('阶段 A 有效条件:', sampler.conditions, '少于两个身份的跳过条件:', sampler.skipped)
    model = DeepIdentityText(_read_clip_checkpoint(clip), len(pid_map), train_id_tokens=True).to(device)
    model.initialize_context()
    print('M2-3 文本: FP32、math attention、TF32 关闭；初始化/锚点设备:', device, flush=True)
    stage = cfg.PROMPT_STAGE
    token_path = output + '.stage_a.pt'
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    cache_hash = file_hash(cache_path)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=str(Path(__file__).resolve().parents[2]), text=True).strip()
    if args.export_only:
        artifact = torch.load(token_path, map_location='cpu', weights_only=False)
        if (artifact['stage_config'] != dict(stage) or artifact['seed'] != seed or
                artifact['updates'] != stage.MAX_UPDATES or
                artifact['feature_cache_sha256'] != cache_hash or artifact['signature'] != model.signature):
            raise ValueError('阶段 A token 文件与当前缓存、模板或训练设置不一致')
        with torch.no_grad():
            model.id_bank.copy_(artifact['id_tokens'])
        updates = artifact['updates']
        artifact['export_code_commit'] = commit
        artifact['export_device'] = str(device)
        artifact['export_torch_version'] = str(torch.__version__)
    else:
        # Fail before 10000 updates if this device/operator combination is invalid.
        _, initial_error = model.check_initial_text()
        print('训练前 t(C0)=t0 检查通过，最大绝对误差:', initial_error, flush=True)
        model.train()
        optimizer = torch.optim.Adam([model.id_bank], lr=stage.LEARNING_RATE, weight_decay=stage.WEIGHT_DECAY)
        features = cache['features'].float().to(device)
        updates = 0
        while updates < stage.MAX_UPDATES:
            condition, ids, rows = sampler.sample(stage.IDENTITIES_PER_STEP, stage.IMAGES_PER_ID)
            step = updates + 1
            if stage.WARMUP_UPDATES and step <= stage.WARMUP_UPDATES:
                lr = stage.LEARNING_RATE * step / stage.WARMUP_UPDATES
            else:
                progress = (step - stage.WARMUP_UPDATES) / max(1, stage.MAX_UPDATES - stage.WARMUP_UPDATES)
                lr = stage.LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad()
            loss = prompt_loss(features[rows.to(device)], model.prompt_forward(ids.to(device)),
                               stage.IMAGES_PER_ID, stage.TEMPERATURE)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite prompt loss; no final artifact is exported')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.id_bank], 1.0)
            optimizer.step()
            updates += 1
            if updates % 1000 == 0 or updates == stage.MAX_UPDATES:
                print('阶段 A update={} loss={:.5f} lr={:.7f}'.format(updates, loss.item(), lr), flush=True)
        artifact = {'variant': 'deep_text_c0_control', 'implementation': 'm2-3-audit-v2',
                    'base_commit': 'a8349abeb5fb62b5f7d233e33202d1ef1d113b42',
                    'code_commit': commit, 'config': cfg.dump(),
                    'prompt_device': str(device), 'torch_version': str(torch.__version__),
                    'text_numerics': 'fp32-math-attention-tf32-off',
                    'actual_iterations': updates, 'optimizer_updates': updates, 'amp_skips': 0,
                    'id_tokens': model.id_bank.detach().cpu(), 'signature': model.signature,
                    'updates': updates, 'stage_config': dict(stage), 'seed': seed,
                    'metadata': cache['metadata'], 'feature_cache_sha256': cache_hash,
                    'condition_updates': sampler.counts, 'skipped_conditions': sampler.skipped}
        # Keep completed training even if anchor validation or export fails.
        torch.save(artifact, token_path)
        print('已保存阶段 A 最终 token:', token_path, flush=True)
    model.eval()
    anchors, error = model.check_initial_text()
    artifact.update({'anchors': anchors, 'context_init': model.context_init.cpu(),
                     'init_text_max_abs_error': error,
                     'text_numerics': 'fp32-math-attention-tf32-off'})
    torch.save(artifact, output)
    print('已导出最终 prompt 工件:', output, '更新数:', updates, 't(C0)=t0 最大误差:', error)

if __name__ == '__main__':
    main()
