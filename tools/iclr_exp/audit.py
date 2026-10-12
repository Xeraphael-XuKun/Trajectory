"""Fixed clean training panel; diagnostics do not update BN or consume RNG."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.m2.artifacts import rng_state, restore_rng


@contextmanager
def isolated_eval(model):
    rng = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)


def build_panel(c, train_data):
    root = Path(c.DATASETS.ROOT_DIR) / (c.DATASETS.SUBDIR or 'WHU-MARS')
    groups, platforms = {}, {}
    for m, name in enumerate(c.DATASETS.MODALITIES):
        for path, pid, camera, _ in train_data[name]:
            raw_pid = int(Path(path).name.split('_')[0])
            groups.setdefault((raw_pid, m, camera), []).append((path, pid))
            platforms.setdefault(raw_pid, set()).add(int(camera in c.DATASETS.AERIAL_CAMS))
    paired = sorted(y for y, v in platforms.items() if v == {0, 1})
    ground = sorted(y for y, v in platforms.items() if v == {0})
    gen = torch.Generator().manual_seed(c.ICLR_EXP.AUDIT.SEED)
    def choose(values, count):
        return [values[i] for i in torch.randperm(len(values), generator=gen)[:count].tolist()]
    selected_p = choose(paired, c.ICLR_EXP.AUDIT.PAIRED_ID_COUNT)
    selected_g = choose(ground, c.ICLR_EXP.AUDIT.GROUND_ONLY_ID_COUNT)
    selected = set(selected_p + selected_g)
    rows = []
    for (raw, m, cam), candidates in sorted(groups.items()):
        if raw not in selected:
            continue
        candidates = sorted(candidates)
        path, pid = candidates[int(torch.randint(len(candidates), (1,), generator=gen))]
        rows.append(dict(relative_path=Path(path).relative_to(root).as_posix(),
            raw_pid=raw, pid=pid, modality=m, camera=cam,
            platform=int(cam in c.DATASETS.AERIAL_CAMS), paired=raw in selected_p))
    return dict(seed=c.ICLR_EXP.AUDIT.SEED, paired_ids=selected_p,
                ground_only_ids=selected_g, rows=rows)


def summary(values):
    x = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    return dict(mean=x.mean().item(), std=x.std(unbiased=False).item(),
                median=x.median().item(), p95=torch.quantile(x, .95).item())


@torch.no_grad()
def run_audit(c, model, panel, epoch, initial_projection, initial_anchors):
    from PIL import Image
    from torchvision import transforms as T
    from loss.iclr_exp import tgva_logits
    root = Path(c.DATASETS.ROOT_DIR) / (c.DATASETS.SUBDIR or 'WHU-MARS')
    transform = T.Compose([T.Resize(c.INPUT.SIZE_TEST), T.ToTensor(),
                          T.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD)])
    device = next(model.parameters()).device
    output = dict(epoch=epoch, input='deterministic_clean_train', rows=panel['rows'])
    on_all, off_all, h_on_all, h_off_all, rho_all = [], [], [], [], {}
    with isolated_eval(model):
        anchors = model.iclr_tgva()
        for start in range(0, len(panel['rows']), c.ICLR_EXP.AUDIT.BATCH_SIZE):
            rows = panel['rows'][start:start + c.ICLR_EXP.AUDIT.BATCH_SIZE]
            images = []
            for row in rows:
                with Image.open(root / row['relative_path']) as image:
                    images.append(transform(image.convert('RGB')))
            x = torch.stack(images).to(device)
            pending, ratios, handles = {}, {}, []
            def capture(_, args, result):
                pending[int(args[0])] = result.detach()
            handles.append(model.base.token_trajectory.register_forward_hook(capture))
            def before(layer):
                def hook(_, args):
                    correction = pending[layer].double()
                    previous = args[0].detach().double() - correction
                    ratios[layer] = torch.stack([
                        correction.flatten(1).norm(dim=1) / previous.flatten(1).norm(dim=1).clamp_min(1e-12),
                        correction[:, 0].norm(dim=1) / previous[:, 0].norm(dim=1).clamp_min(1e-12),
                        correction[:, 1:].flatten(1).norm(dim=1) / previous[:, 1:].flatten(1).norm(dim=1).clamp_min(1e-12)], dim=1).cpu()
                return hook
            for layer in range(1, 12):
                handles.append(model.base.blocks[layer].register_forward_pre_hook(before(layer)))
            try:
                on = model.base(x)
            finally:
                for handle in handles:
                    handle.remove()
            off = model.base(x, trajectory_gate=x.new_zeros(x.shape[0]))
            h_on, h_off = tgva_logits(on, off, model.base.clip_proj, anchors, model.iclr_tgva.logit_scale)
            on_all.append(on.float().cpu()); off_all.append(off.float().cpu())
            h_on_all.append(h_on.cpu()); h_off_all.append(h_off.cpu())
            for layer, value in ratios.items():
                rho_all.setdefault(layer, []).append(value)
        on, off = torch.cat(on_all), torch.cat(off_all)
        h_on, h_off = torch.cat(h_on_all), torch.cat(h_off_all)
        mods = torch.tensor([r['modality'] for r in panel['rows']])
        views = torch.tensor([r['platform'] for r in panel['rows']])
        tasks = []
        for m in range(3):
            for v in range(2):
                mask = (mods == m) & (views == v)
                if not mask.any():
                    continue
                for b, logits in (('on', h_on), ('off', h_off)):
                    target = 2 * m + (v if b == 'off' else 0)
                    logp = logits[mask].log_softmax(-1)
                    tasks.append(dict(modality=m, platform=v, branch=b, count=int(mask.sum()),
                        target=target, target_accuracy=float((logp.argmax(-1) == target).float().mean()),
                        observed_accuracy=float((logp.argmax(-1) == 2*m+v).float().mean()),
                        modality_accuracy=float((logp.argmax(-1)//2 == m).float().mean()),
                        platform_accuracy=float((logp.argmax(-1)%2 == v).float().mean()),
                        ce=float(-logp[:, target].mean()), entropy=float(-(logp.exp()*logp).sum(-1).mean())))
        index = torch.arange(len(mods))
        delta = (h_on[index, 2*mods]-h_on[index, 2*mods+1]) - (h_off[index, 2*mods]-h_off[index, 2*mods+1])
        output['tasks'] = tasks
        output['on_off_cosine'] = F.cosine_similarity(on, off).tolist()
        output['on_off_relative_distance'] = ((on-off).norm(dim=1)/off.norm(dim=1).clamp_min(1e-12)).tolist()
        output['ground_minus_aerial_log_odds_change'] = delta.tolist()
        projection = model.base.clip_proj.detach().float().cpu()
        z_on, z_off = F.normalize(on @ projection, dim=-1), F.normalize(off @ projection, dim=-1)
        output['joint_on_off_cosine'] = F.cosine_similarity(z_on, z_off).tolist()
        output['projection_relative_drift'] = float((projection-initial_projection).norm()/initial_projection.norm())
        anchors = anchors.cpu()
        output['anchor_cosine'] = (anchors @ anchors.t()).tolist()
        output['anchor_initial_cosine'] = (anchors*initial_anchors).sum(-1).tolist()
        p = model.iclr_tgva.prompts
        output['token_norms'] = {name: getattr(p, name).detach().double().norm(dim=-1).cpu().tolist()
                                 for name in ('view_ctx', 'modality_ctx', 'shared_ctx')}
        gain = model.base.token_trajectory.gain.detach().double().cpu().flatten(1)
        output['gain'] = dict(rms=gain.square().mean(-1).sqrt().tolist(), maxabs=gain.abs().amax(-1).tolist())
        output['correction_ratios'] = {}
        for layer, values in rho_all.items():
            values = torch.cat(values)
            entries = []
            for m in range(3):
                for v in range(2):
                    mask = (mods == m) & (views == v)
                    if mask.any():
                        entries.append(dict(modality=m, platform=v, count=int(mask.sum()),
                            all=summary(values[mask, 0]), cls=summary(values[mask, 1]), patch=summary(values[mask, 2])))
            output['correction_ratios'][str(layer+1)] = entries
    os.makedirs(os.path.join(c.OUTPUT_DIR, 'audit'), exist_ok=True)
    with open(os.path.join(c.OUTPUT_DIR, 'audit', 'epoch%d.json' % epoch), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    return output
