import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from loss.twin_align import twin_align_loss
from loss.twin_infonce import twin_infonce_loss
from loss.text_align import (AERIAL, GROUND, anchor_cos,
                             text_align_loss, view_logits)
from loss.vpr_losses import (vpr_cross_spectral_displacement_loss,
                             vpr_view_text_loss)
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F


def PCA(feat, target, num_modalities, group_size=4):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    return loss, ids, g_cent


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        P = bank.prototypes.clone()
        norms = P.norm(dim=1)
        need = norms[ids] < 1e-6
        if need.any():
            P[ids[need]] = g_cent[need]
        P = F.normalize(P, dim=1)

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return loss


def feature_geometry(feat, n_modalities, group_size):
    """d_same / d_cross / d_diffpid on the 768-d retrieval feature, per batch.

    The reason this is here at all: every other number in the text diagnostic
    lives in the 512-d CLIP space and is measured against prompts that are
    themselves learning, so in principle the anchors could have moved instead
    of the images.  These three touch no prompt and no projection -- they are
    the same quantities diag/analyse_modality.py reports on the test set, and
    they are what the correction actually has to change:

        d_cross(RGB, Thermal)   1.125 at the start   ->  1.019 would match RGB<->IR
        d_cross(RGB, IR)        1.019               ->  must not get worse
        d_diffpid               1.273               ->  must not come down with it

    (Those are ratios to d_same, which is why d_same is returned too.)

    Position is the label here: PKMSampler yields modality-major rows with each
    identity occupying `group_size` consecutive slots, which PCA above already
    relies on.  A batch that does not divide evenly is skipped rather than
    guessed at.
    """
    n = feat.shape[0]
    per = n // n_modalities
    if n % n_modalities or per % group_size or per // group_size < 2:
        return None
    with torch.no_grad():
        f = F.normalize(feat.float(), dim=-1)
        d = torch.cdist(f, f)
        idx = torch.arange(n, device=f.device)
        mod = idx // per
        pid = (idx % per) // group_size
        same_pid = pid[:, None] == pid[None, :]
        same_mod = mod[:, None] == mod[None, :]
        off = ~torch.eye(n, dtype=torch.bool, device=f.device)

        out = {'d_same': d[same_pid & same_mod & off].mean().item(),
               'd_diffpid': d[~same_pid & same_mod].mean().item(),
               'd_cross': {}}
        for a in range(n_modalities):
            for b in range(a + 1, n_modalities):
                sel = same_pid & (mod[:, None] == a) & (mod[None, :] == b)
                out['d_cross'][(a, b)] = d[sel].mean().item()
    return out


def ce_split_target(target_rep, camids, n_modalities, aerial_cams,
                    split_view, split_modality, modality_groups=None):
    """`pid` -> `pid * slots + slot`, where slot encodes the view / spectrum.

    Returns (labels, slots).  With both splits off it returns `target_rep`
    unchanged and slots = 1, so a run that does not ask for a split is bit for
    bit the old one.

    `modality_groups` is parallel to DATASETS.MODALITIES and says which slot
    each spectrum lands in; None (the default) gives every spectrum its own,
    which is whu_ce_mod.  [0, 0, 1] merges RGB and IR -- the pair whu_ce_mod
    was measured to have broken while gaining nothing from separating -- and
    leaves Thermal, the split that paid, on its own.

    Row i of the batch belongs to spectrum `i // per_modality` -- the
    modality-major layout train_collate_fn produces, the same convention
    `_text_subset` and `twin_infonce_loss` rely on.  The viewpoint comes from
    the camera id, which is why DATASETS.AERIAL_CAMS has to be non-empty for
    the view split to mean anything; do_train refuses that combination rather
    than silently mapping every image to "ground".

    The slot is a mixed-radix number so the two splits compose: with both on,
    3 spectra x 2 views = 6 slots per identity.
    """
    n = target_rep.shape[0]
    per = n // n_modalities
    device = target_rep.device
    slot = torch.zeros(n, dtype=torch.long, device=device)
    slots = 1
    if split_modality:
        row_mod = torch.arange(n, device=device) // per
        if modality_groups is None:
            group, n_groups = row_mod, n_modalities
        else:
            gmap = torch.as_tensor(list(modality_groups), dtype=torch.long,
                                   device=device)
            group, n_groups = gmap[row_mod], int(gmap.max()) + 1
        slot = slot * n_groups + group
        slots *= n_groups
    if split_view:
        cams = torch.cat([c.to(device) for c in camids], dim=0)
        slot = slot * 2 + torch.isin(cams, aerial_cams.to(device)).long()
        slots *= 2
    return target_rep * slots + slot, slots


def modality_align(aux):
    """The 2 x n_spectrum spectrum supervision.  Returns (loss, stats).

                     before (no delta)        after (delta applied)
        RGB image      -> "color"               -> "color"
        infrared       -> "infrared"            -> "color"
        thermal        -> "thermal"             -> "color"

    The after column is per spectrum, not one shared target.  With
    MODEL.TEXT_MODALITY_TARGETS [0, 1, 0] it reads

        RGB image      -> "color"               -> "color"
        infrared       -> "infrared"            -> "infrared"
        thermal        -> "thermal"             -> "color"

    i.e. only thermal is asked to move.  That mirrors CE_MODALITY_GROUPS for
    the same measured reason -- RGB<->IR was already free before anything was
    split, so aligning it again is redundant work on the one pair that a
    heavier hand is known to break.  `aux['target_rows']` None reproduces the
    single-target form exactly, which is what keeps whu_mcam comparable.

    Structurally `view_align` with the axis swapped from viewpoint to spectrum:
    every cell is a cross entropy over the same anchors, and the only cells
    asking for a change are the non-reference spectra after the delta.  Two
    differences from the view form, both forced by the axis rather than chosen:

    * every image in the batch is supervised, not one third.  The view form
      asked ~17 of 192 images to change; here 128 of 192 do.
    * the "before" cells are the anti-collapse guard.  Without them the
      cheapest solution is to collapse the three anchors together so that
      everything reads as the reference spectrum and the loss vanishes with no
      feature having moved.  They are computed on the UNCORRECTED features, so
      satisfying them forces the backbone to keep the spectra separable while
      the delta makes them read alike.
    """
    proj, text, scale = aux['proj'], aux['text'], aux['logit_scale']
    modality, tgt = aux['modality'], aux['target_row']
    to_clip = lambda f: f.float() @ proj.float()

    cos_b = anchor_cos(to_clip(aux['feat_before']), text)     # [N, n_spectrum]
    cos_a = anchor_cos(to_clip(aux['feat_after']), text)

    n_spectrum = text.shape[0]
    rows = aux.get('target_rows')
    if rows is None:
        rows = [int(tgt)] * n_spectrum
    rows = torch.as_tensor(list(rows), dtype=torch.long, device=modality.device)
    after_tgt = rows[modality]                     # [N], each row's own target

    # before: each spectrum reads as itself  (the anti-collapse guard)
    loss = F.cross_entropy(cos_b * scale, modality)
    # after: each spectrum reads as the anchor assigned to it.  Where that is
    # its own anchor the cell says "stay put"; elsewhere it is the correction
    # being asked for.
    loss = loss + F.cross_entropy(cos_a * scale, after_tgt)

    stats = {'n_total': int(modality.numel()),
             'n_moved': int((modality != after_tgt).sum()), 'target_row': int(tgt),
             'target_rows': [int(r) for r in rows.tolist()],
             'cos_before': [], 'cos_after': [],
             'acc_before': [], 'acc_after': []}
    for m in range(n_spectrum):
        sel = modality == m
        if int(sel.sum()) == 0:
            for k in ('cos_before', 'cos_after', 'acc_before', 'acc_after'):
                stats[k].append(float('nan'))
            continue
        # This spectrum's OWN target, so cos_before and cos_after are measured
        # against the same anchor and their difference is readable.  Identical
        # to the old reading whenever every row targets the reference.
        t = int(rows[m])
        with torch.no_grad():
            stats['cos_before'].append(cos_b[sel, t].mean().item())
            stats['cos_after'].append(cos_a[sel, t].mean().item())
            # before: does this spectrum still read as ITSELF (guard holding)
            stats['acc_before'].append(
                (cos_b[sel].argmax(dim=1) == m).float().mean().item())
            # after: does it now read as its target (correction landed).
            # Both saturate -- that is accepted here, and the cosines above are
            # what stays readable once they do.
            stats['acc_after'].append(
                (cos_a[sel].argmax(dim=1) == t).float().mean().item())
    # The guard reading, unchanged: how far the REFERENCE spectrum itself moved
    # against its own anchor.  It should be near zero however the other rows
    # are targeted.
    stats['drift'] = stats['cos_after'][tgt] - stats['cos_before'][tgt]
    return loss, stats


def view_align(aux):
    """The 2x2 view supervision.  Returns (loss, stats-for-the-log).

    Both features go through CLIP's `visual.proj` into the 512-d joint space --
    the 768-d CLS lives in the backbone's own space and is not comparable to a
    text vector at all.

    With modality anchors the anchor row is (modality, view), so every sample
    carries its own target and its own (aerial, ground) pair for the margin;
    comparing a thermal feature against the RGB anchors would be measuring the
    wrong direction.  Without them `modality` is all zeros and this reduces to
    the two-anchor case exactly.

    The `margin` entries are the point of the diagnostic: they say how far each
    group sits towards the ground anchor, so `push` (how far the delta moved
    the aerial images) and `drift` (how far it moved the ground ones, which
    should be ~0) are readable straight off the log.  A run where the delta
    never moved and a run where the idea does not work produce the same mAP;
    only these two numbers tell them apart.
    """
    proj, text, scale = aux['proj'], aux['text'], aux['logit_scale']
    is_aerial, modality = aux['is_aerial'], aux['modality']
    n_view = aux['n_view']
    to_clip = lambda f: f.float() @ proj.float()

    def rows(view_idx, sel):
        """anchor row per selected sample, for a fixed target view."""
        return modality[sel] * n_view + view_idx

    def pairs(sel):
        """[N, 2] of (aerial row, ground row) in each sample's own modality."""
        base = modality[sel] * n_view
        return torch.stack([base + AERIAL, base + GROUND], dim=1)

    groups = {
        'A_before': (aux['feat_before'], is_aerial, AERIAL),
        'A_after': (aux['feat_after'], is_aerial, GROUND),
        'G_before': (aux['feat_before'], ~is_aerial, GROUND),
        'G_after': (aux['feat_after'], ~is_aerial, GROUND),
    }
    total, stats = 0.0, {}
    for name, (feat, sel, view_idx) in groups.items():
        loss, acc, margin = text_align_loss(
            to_clip(feat[sel]), text, rows(view_idx, sel), scale, pairs(sel))
        total = total + loss
        stats['acc_' + name] = acc.item()
        stats['margin_' + name] = margin.item()
    stats['push'] = stats['margin_A_after'] - stats['margin_A_before']
    stats['drift'] = stats['margin_G_after'] - stats['margin_G_before']
    stats['n_aerial'] = int(is_aerial.sum())
    stats['n_total'] = int(is_aerial.numel())

    # Per-modality push.  With one modality supervised this is the same number
    # again; with three it answers the question the modality slot was added
    # for -- whether the correction the delta learns is shared across spectra
    # or only ever applies to the one CLIP understands.
    if aux['n_modality'] > 1:
        per = []
        for m in range(aux['n_modality']):
            sel = is_aerial & (modality == m)
            if int(sel.sum()) == 0:
                per.append(float('nan')); continue
            b = text_align_loss(to_clip(aux['feat_before'][sel]), text,
                                rows(AERIAL, sel), scale, pairs(sel))[2].item()
            a = text_align_loss(to_clip(aux['feat_after'][sel]), text,
                                rows(GROUND, sel), scale, pairs(sel))[2].item()
            per.append(a - b)
        stats['push_per_modality'] = per
    return total, stats


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    logger.info('start training')
    relation_bank = None
    if cfg.M2.ENABLED:
        from loss.clip_relation import identity_modality_centers, relation_kl
        if cfg.M2.VARIANT != 'clip_relation_distill' or cfg.M2.RELATION_READOUT != 'after':
            raise ValueError('M2-4 requires clip_relation_distill and post-BN readout')
        if not cfg.M2.RELATION_REMOVE_SAME_ID or not cfg.M2.RELATION_ALL_MODAL_PAIRS:
            raise ValueError('M2-4 first configuration requires diagonal removal and all nine pairs')
        from tools.m2_cache import validate_artifact
        artifact = torch.load(cfg.M2.RELATION_BANK, map_location='cpu', weights_only=False)
        validate_artifact(artifact, cfg)
        relation_bank = artifact['relation'].to(device).detach()
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]
    meter_text = AverageMeter()
    text_weight = cfg.SOLVER.TEXT_LOSS_WEIGHT if cfg.MODEL.TEXT_ALIGN else 0.0
    last_text_stats = None

    meter_vpr_text = AverageMeter()
    meter_vpr_csd = AverageMeter()
    vpr_text_weight = (float(cfg.SOLVER.VPR_VIEW_TEXT_WEIGHT)
                       if cfg.MODEL.VPR and cfg.MODEL.VPR_VIEW_TEXT else 0.0)
    vpr_csd_weight = (float(cfg.SOLVER.VPR_CSD_WEIGHT)
                      if cfg.MODEL.VPR and cfg.MODEL.VPR_CSD else 0.0)
    vpr_aux_only = bool(cfg.SOLVER.VPR_AUX_RESIDUAL_ONLY)
    last_vpr_text_stats = last_vpr_csd_stats = None
    if vpr_csd_weight > 0:
        if not cfg.DATALOADER.SYNC_FRAMES:
            raise ValueError('VPR_CSD_WEIGHT > 0 needs SYNC_FRAMES True')
        if not cfg.DATALOADER.TIE_AUGMENTATION:
            raise ValueError(
                'VPR CSD needs TIE_AUGMENTATION True so crop/flip agree across spectra')
    if (vpr_text_weight > 0 or vpr_csd_weight > 0) and cfg.MODEL.DIST_TRAIN:
        raise NotImplementedError(
            'VPR synchronized tuples and residual-only auxiliary gradients are '
            'currently single-GPU; DATALOADER.SYNC_FRAMES already has the same '
            'restriction')

    meter_twin = AverageMeter()
    twin_weight = cfg.SOLVER.TWIN_LOSS_WEIGHT if cfg.MODEL.MOD_DELTA else 0.0
    last_twin_stats = None
    meter_tnce = AverageMeter()
    tnce_weight = cfg.SOLVER.TNCE_WEIGHT
    tnce_warmup = int(cfg.SOLVER.TNCE_WARMUP_EPOCHS)
    last_tnce_stats = None
    if (twin_weight > 0 or tnce_weight > 0) and not cfg.DATALOADER.SYNC_FRAMES:
        # Without frame synchronisation rows i and i+per_modality are the same
        # person at unrelated moments, from possibly different cameras.  The
        # loss would then be pulling together two different poses and calling
        # the difference "spectrum", which is a different experiment wearing
        # this one's name.  Fail here rather than produce a plausible number.
        raise ValueError(
            'SOLVER.TWIN_LOSS_WEIGHT / TNCE_WEIGHT > 0 need DATALOADER.SYNC_FRAMES '
            'True -- the twin target is the SAME capture, and the default sampler '
            'draws each modality independently')

    ce_split_view = bool(cfg.MODEL.CE_SPLIT_VIEW)
    ce_split_mod = bool(cfg.MODEL.CE_SPLIT_MODALITY)
    # None, not the equivalent [0, 1, 2], so that every run recorded before the
    # grouping existed takes literally the same branch it took then.  The list
    # itself was validated by ce_slot_count when the classifier was built.
    ce_groups = list(cfg.MODEL.CE_MODALITY_GROUPS) or None
    ce_aerial = torch.tensor(sorted(set(int(c) for c in cfg.DATASETS.AERIAL_CAMS)),
                             dtype=torch.long)
    if ce_split_view and ce_aerial.numel() == 0:
        # Every image would count as ground, the view split would be a no-op,
        # and the run would look exactly like the unsplit one in every line of
        # the log.  All the WHU configs ship AERIAL_CAMS empty, so this is the
        # likely mistake rather than an exotic one.
        raise ValueError(
            'MODEL.CE_SPLIT_VIEW needs DATASETS.AERIAL_CAMS -- without it every '
            'image is treated as ground and the split does nothing at all')

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        metric=cfg.TEST.METRIC,
        aerial_cams=cfg.DATASETS.AERIAL_CAMS,
    )
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    print(feat_dim, num_classes)
    if use_aux_modules:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=cfg.MODEL.GPD_MOMENTUM,
            device=device,
        )

    group_size = cfg.DATALOADER.NUM_INSTANCE

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        meter_text.reset()
        meter_vpr_text.reset()
        meter_vpr_csd.reset()
        meter_twin.reset()
        meter_tnce.reset()
        # Linear ramp, per epoch rather than per iteration: the point is to let
        # CE and the triplet settle first, and a smoother curve inside an epoch
        # would not change that while making the log harder to line up with the
        # baseline's.  Epoch 1 gets exactly 0, so it must reproduce the
        # baseline -- which is the cheapest check that nothing else in this
        # config is doing damage.
        tnce_now = (tnce_weight if tnce_warmup <= 0 else
                    tnce_weight * min(1.0, (epoch - 1) / float(tnce_warmup)))
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        n_iter = 0
        for n_iter, (imgs, vid, camids) in enumerate(train_loader):
            # No-op unless the scheduler counts updates rather than epochs
            # (SOLVER.WARMUP_ITERS > 0); get_update_values returns None in
            # epoch mode and Scheduler.step_update then leaves the groups
            # alone, exactly as scheduler.step(epoch) does in update mode.
            scheduler.step_update((epoch - 1) * len(train_loader) + n_iter)

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)

            with amp.autocast(enabled=True):
                out = model(imgs, target, camids)
                aux = out[3] if len(out) > 3 else None
                cls_score, global_feat, feat = out[0], out[1], out[2]

                target_ce = None
                if ce_split_view or ce_split_mod:
                    target_ce, _slots = ce_split_target(
                        target_rep, camids, num_modalities, ce_aerial,
                        ce_split_view, ce_split_mod, ce_groups)
                loss, il, tl = loss_fn(cls_score, global_feat, target_rep,
                                       target_ce=target_ce)
                if relation_bank is not None:
                    with amp.autocast(enabled=False):
                        mods = torch.arange(num_modalities, device=feat.device).repeat_interleave(len(target))
                        centers, ids, valid = identity_modality_centers(feat, target_rep, mods)
                        rel_loss, rel_stats = relation_kl(
                            centers, relation_bank[ids][:, ids],
                            cfg.M2.STUDENT_TEMPERATURE, cfg.M2.TEACHER_TEMPERATURE,
                            valid=valid, return_stats=True)
                        loss = loss + cfg.M2.RELATION_WEIGHT * rel_loss
                    if n_iter % log_period == 0:
                        logger.info('M2-4 relation_loss=%s stats=%s', rel_loss.item(), rel_stats)
                vpr_aux_total = loss.new_zeros(())

                if aux is not None and aux.get('kind') == 'vpr':
                    if vpr_text_weight > 0:
                        raw_vpr_text, last_vpr_text_stats = vpr_view_text_loss(aux)
                        vpr_aux_total = (vpr_aux_total +
                                         vpr_text_weight * raw_vpr_text)
                        meter_vpr_text.update(raw_vpr_text.item(),
                                              max(1, aux['text_rows'].sum().item()))
                    if vpr_csd_weight > 0:
                        raw_vpr_csd, last_vpr_csd_stats = \
                            vpr_cross_spectral_displacement_loss(aux)
                        vpr_aux_total = (vpr_aux_total +
                                         vpr_csd_weight * raw_vpr_csd)
                        meter_vpr_csd.update(raw_vpr_csd.item(),
                                             max(1, last_vpr_csd_stats['n_valid']))

                if (aux is not None and aux.get('kind') != 'vpr'
                        and text_weight > 0):
                    if aux.get('target') == 'modality':
                        loss_text, text_stats = modality_align(aux)
                    else:
                        loss_text, text_stats = view_align(aux)
                    loss = loss + text_weight * loss_text
                    meter_text.update(loss_text.item(), aux['feat_after'].shape[0])
                    last_text_stats = text_stats

                if twin_weight > 0:
                    per_modality = imgs[0].shape[0]
                    # ref_index defaults to 0, which is the reference
                    # modality: make_model refuses any MOD_DELTA_REF that is not
                    # DATASETS.MODALITIES[0], so block 0 of this modality-major
                    # batch is always the frame the others are corrected into.
                    loss_twin, twin_cos = twin_align_loss(
                        global_feat, num_modalities, per_modality)
                    loss = loss + twin_weight * loss_twin
                    meter_twin.update(loss_twin.item(), target_rep.shape[0])
                    last_twin_stats = twin_cos
                if tnce_now > 0:
                    per_modality = imgs[0].shape[0]
                    loss_tnce, tnce_stats = twin_infonce_loss(
                        global_feat, target, num_modalities, per_modality,
                        cfg.SOLVER.TNCE_TAU)
                    loss = loss + tnce_now * loss_tnce
                    meter_tnce.update(loss_tnce.item(), target_rep.shape[0])
                    last_tnce_stats = tnce_stats

                meter_ls[0].update(il.item(), target_rep.shape[0])
                meter_ls[1].update(tl.item(), target_rep.shape[0])

                if use_aux_modules:
                    loss_pca, ids, g_cent = PCA(
                        global_feat,
                        target,
                        num_modalities=num_modalities,
                        group_size=group_size,
                    )
                    meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                    loss_gpd = GPD(
                        global_feat,
                        target,
                        proto_bank,
                        ids,
                        g_cent,
                        num_modalities=num_modalities,
                        tau=0.03,
                    )
                    meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                    if 'pca' in loss_type:
                        loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                    if 'gpd' in loss_type:
                        loss += loss_gpd

            if ((vpr_text_weight > 0 or vpr_csd_weight > 0)
                    and vpr_aux_only):
                # Main CE/Triplet follows the ordinary full-model path.  The
                # two VPR auxiliaries contribute gradients ONLY to VPR.  The
                # gradients below are already GradScaler-scaled, so unscale_
                # handles them together with the ordinary parameter grads.
                scaler.scale(loss).backward(retain_graph=True)
                vpr_params = [p for p in model_meta.base.vpr.parameters()
                              if p.requires_grad]
                aux_grads = torch.autograd.grad(
                    scaler.scale(vpr_aux_total), vpr_params,
                    allow_unused=True)
                for parameter, grad in zip(vpr_params, aux_grads):
                    if grad is not None:
                        parameter.grad = (grad if parameter.grad is None
                                          else parameter.grad + grad)
                display_loss = loss.detach() + vpr_aux_total.detach()
            else:
                total_loss = loss + vpr_aux_total
                scaler.scale(total_loss).backward()
                display_loss = total_loss.detach()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if use_aux_modules:
                with torch.no_grad():
                    proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)

            # Against the CE's OWN label space.  With MODEL.CE_SPLIT_* the
            # classifier predicts `pid * slots + slot` while target_rep is the
            # raw pid, so comparing the two reports ~0 however well the head is
            # doing -- measured on whu_ce_mod, Acc read 0.001 while the cross
            # entropy itself had fallen from ln(1500)=7.31 to 2.36.
            acc = (cls_score.max(1)[1] ==
                   (target_rep if target_ce is None else target_ce)).float().mean()
            loss_meter.update(display_loss.item(), target.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                # Read the lr the optimizer is actually using.  Asking the
                # scheduler for _get_lr(epoch) reports the wrong number the
                # moment the schedule is indexed by updates instead of epochs,
                # and it only ever showed the first group -- which with a
                # two-tier setup is the pretrained tower, not "base lr".
                lrs = [g['lr'] for g in optimizer.param_groups]
                lr_str = ('{:.2e}'.format(max(lrs)) if min(lrs) == max(lrs)
                          else '{:.2e} (pretrained {:.2e})'.format(max(lrs), min(lrs)))
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        lr_str,
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')

                # Unconditionally, not only under the text / twin losses.  It
                # used to live inside those branches, so a plain run printed
                # nothing and the first read on whether a change moved the
                # feature geometry had to wait for the run to finish, the
                # checkpoint to survive, and a feature dump to be scheduled.
                #
                # It is a DIFFERENT quantity from the one
                # diag/analyse_modality.py reports and the two must not be
                # compared: this one is batch-internal, over 192 augmented
                # images, with whatever cameras the sampler happened to draw.
                # The offline one runs on 93,609 clean gallery images and
                # requires the pair to come from DIFFERENT cameras, which is
                # what the metric scores.  Read this for the trend inside a
                # run, never as a level against the test set.
                #
                # Cost: one cdist on 192x768 under no_grad, once every
                # LOG_PERIOD iterations.  Against a ViT-B forward over 192
                # images that is not measurable.
                geom = feature_geometry(global_feat, num_modalities, group_size)
                if geom:
                    names = list(cfg.DATASETS.MODALITIES)
                    d = geom['d_same']
                    logger.info(
                        'Epoch[{}]   768d d_same={:.4f}  {}  d_diffpid={:.3f}'
                        .format(epoch, d,
                                '  '.join('d_cross({},{})={:.3f}'.format(
                                    names[a], names[b], v / d)
                                    for (a, b), v in geom['d_cross'].items()),
                                geom['d_diffpid'] / d))
                if last_vpr_text_stats is not None:
                    s = last_vpr_text_stats
                    logger.info(
                        'Epoch[{}] VPR-view-text: raw={:.4f} x {:.1f}; '
                        'dir={:.4f} orth={:.4f} ground={:.4f}'.format(
                            epoch, meter_vpr_text.avg, vpr_text_weight,
                            s['direction'], s['orthogonal'], s['ground_keep']))
                    logger.info(
                        'Epoch[{}]   cos(direction)={:+.3f} |dv|={:.4f} '
                        '|dt|={:.4f} nA={} nG={}'.format(
                            epoch, s['direction_cos'], s['displacement_norm'],
                            s['target_norm'], s['n_aerial'], s['n_ground']))
                if last_vpr_csd_stats is not None:
                    vals = last_vpr_csd_stats['per_modality']
                    logger.info(
                        'Epoch[{}] VPR-CSD: raw={:.4f} x {:.1f}; n={} {}'.format(
                            epoch, meter_vpr_csd.avg, vpr_csd_weight,
                            last_vpr_csd_stats['n_valid'],
                            '  '.join('{}={:.4f}'.format(n, v) for n, v in
                                     zip(cfg.DATASETS.MODALITIES, vals))))
                if last_text_stats is not None and 'acc_after' in last_text_stats:
                    s = last_text_stats
                    names = list(cfg.DATASETS.MODALITIES)
                    tgt = s['target_row']
                    fmt = lambda vals: '  '.join(
                        '{}={:+.3f}'.format(n, v) for n, v in zip(names, vals))
                    logger.info(
                        'Epoch[{}] Text: loss={:.3f} n={}/{} moved  target={}'.format(
                            epoch, meter_text.avg, s['n_moved'], s['n_total'],
                            names[tgt]))
                    logger.info('Epoch[{}]   cos->{} before  {}'.format(
                        epoch, names[tgt], fmt(s['cos_before'])))
                    logger.info('Epoch[{}]   cos->{} after   {}   drift={:+.3f}'.format(
                        epoch, names[tgt], fmt(s['cos_after']), s['drift']))
                    # Both accuracies saturate; the cosines above are what stays
                    # readable after they do.  acc(before) is the guard -- if it
                    # falls, the backbone is letting the spectra merge, which is
                    # the collapse this design exists to prevent.
                    acc = lambda vals: '  '.join('{}={:.2f}'.format(n, v)
                                                 for n, v in zip(names, vals))
                    logger.info('Epoch[{}]   acc(before,=self)  {}'.format(
                        epoch, acc(s['acc_before'])))
                    logger.info('Epoch[{}]   acc(after,->{})   {}'.format(
                        epoch, names[tgt], acc(s['acc_after'])))
                elif last_text_stats is not None:
                    s = last_text_stats
                    logger.info(
                        'Epoch[{}] Text: loss={:.3f} n_aerial={}/{}'.format(
                            epoch, meter_text.avg, s['n_aerial'], s['n_total']))
                    logger.info(
                        'Epoch[{}]   acc  A前={:.2f} A后={:.2f} G前={:.2f} G后={:.2f}'.format(
                            epoch, s['acc_A_before'], s['acc_A_after'],
                            s['acc_G_before'], s['acc_G_after']))
                    logger.info(
                        'Epoch[{}]   margin A {:+.3f}->{:+.3f} push={:+.3f} | '
                        'G {:+.3f}->{:+.3f} drift={:+.3f}'.format(
                            epoch, s['margin_A_before'], s['margin_A_after'], s['push'],
                            s['margin_G_before'], s['margin_G_after'], s['drift']))
                    if 'push_per_modality' in s:
                        # The question the modality slot was added for: does the
                        # correction transfer across spectra, or only work on the
                        # one CLIP actually understands?
                        logger.info(
                            'Epoch[{}]   push per modality: {}'.format(
                                epoch, ' '.join(
                                    '{}={:+.3f}'.format(n, v) for n, v in
                                    zip(cfg.DATASETS.MODALITIES, s['push_per_modality']))))
                if last_tnce_stats is not None:
                    st = last_tnce_stats
                    names = list(cfg.DATASETS.MODALITIES)
                    fmt = lambda k: '  '.join(
                        '{}={:.4f}'.format(n, v)
                        for n, v in zip(names, st[k]) if v == v)
                    # rank1 first and loss last, deliberately.  The probe
                    # measured that plain centring drops this loss by 57% while
                    # moving rank1 by 0.1 points, so the loss falling is not
                    # evidence of anything and rank1 is.
                    logger.info('Epoch[{}] Twin-NCE: rank1  {}'.format(
                        epoch, fmt('rank1')))
                    logger.info('Epoch[{}]            cos    {}'.format(
                        epoch, fmt('cos')))
                    logger.info(
                        'Epoch[{}]            margin {}   loss={:.4f}   w={:.3f}'
                        .format(epoch, fmt('margin'), meter_tnce.avg, tnce_now))

                if last_twin_stats is not None:
                    twin_cos = last_twin_stats
                    names = list(cfg.DATASETS.MODALITIES)
                    logger.info('Epoch[{}] Twin: loss={:.4f}   {}'.format(
                        epoch, meter_twin.avg,
                        '  '.join('cos({}->twin)={:.4f}'.format(n, c)
                                  for n, c in zip(names, twin_cos)
                                  if c == c)))
                chart_stats = getattr(getattr(model_meta, 'base', model_meta), 'last_chart_stats', None)
                if chart_stats:
                    stats_str = ', '.join(f'{k}={v:.4f}' for k, v in chart_stats.items())
                    logger.info(f'Epoch[{epoch}] Chart: {stats_str}')

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()

            dist_on = (
                cfg.MODEL.DIST_TRAIN
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
            rank = dist.get_rank() if dist_on else 0
            world_sz = dist.get_world_size() if dist_on else 1

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for img, vid, camid, camidt, modid, _ in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img, camids=camidt.to(device), mode=mode)
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        metric=cfg.TEST.METRIC,
        aerial_cams=cfg.DATASETS.AERIAL_CAMS,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for img, vid, camid, camidt, modid, imgpath in val_loader:
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, camids=camidt.to(device), mode=mode)
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
