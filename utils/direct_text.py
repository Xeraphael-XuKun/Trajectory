"""M2-5-DT: fixed P5 text, ordinary C0 forward, direct identity CE only."""
import copy
import hashlib
import json
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from yacs.config import CfgNode

PARENT = 'db8a5c65ba0b7b5f0c62d00d2adb2f5e3bfe1177'
TEXT_SHA = '963ffc875fc5bbce7b6f39cb9ee8760ef94425c4202461aa5be0310bee875a67'
CLIP_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
IMPLEMENTATION = 'm2-5-dt-v1'

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def validate_config(c):
    """Only called for DT; the disabled C0 path has no resource side effects."""
    if not c.M2_DIRECT_TEXT.ENABLED:
        return
    assert not any((c.M2.ENABLED, c.MODEL.TEXT_ALIGN, c.MODEL.VPR, c.MODEL.MOD_DELTA,
        c.MODEL.CE_SPLIT_VIEW, c.MODEL.CE_SPLIT_MODALITY, c.MODEL.SIE_CAMERA, c.MODEL.SIE_VIEW,
        c.MODEL.DIST_TRAIN))
    # This actual M5 parent predates/removes these optional config blocks.
    for legacy in ('C0_AUX','TERMINAL_REPAIR'):
        if hasattr(c,legacy):
            assert not getattr(c,legacy).ENABLED
    assert c.MODEL.TOKEN_TRAJECTORY and c.MODEL.TOKEN_TRAJECTORY_VARIANT == 'dense'
    assert c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX == 0. and c.MODEL.PE_LAYERWISE == 'none'
    assert c.MODEL.TRANSFORMER_TYPE == 'vit_base_clip' and c.MODEL.PRETRAIN_CHOICE == 'imagenet'
    assert not c.MODEL.PE_FREEZE_BASE and c.MODEL.PE_TYPE == 'learnable' and not c.MODEL.LAYER_SCALE
    assert list(c.INPUT.SIZE_TRAIN) == list(c.INPUT.SIZE_TEST) == [256,128]
    assert list(c.MODEL.STRIDE_SIZE) == [16,16]
    assert c.MODEL.IF_LABELSMOOTH == 'off' and c.MODEL.IF_WITH_CENTER == 'no'
    assert c.MODEL.METRIC_LOSS_TYPE == 'triplet' and c.SOLVER.LOSS_TYPE == 'base'
    assert c.MODEL.ID_LOSS_WEIGHT == c.MODEL.TRIPLET_LOSS_WEIGHT == 1.
    assert not any((c.SOLVER.TEXT_LOSS_WEIGHT, c.SOLVER.TWIN_LOSS_WEIGHT, c.SOLVER.TNCE_WEIGHT))
    assert list(c.DATASETS.MODALITIES) == ['RGB', 'IR', 'Thermal']
    assert c.DATASETS.PROTOCOL == 'ALL' and c.DATASETS.NAMES == 'WHU-MARS'
    assert not c.DATALOADER.SYNC_FRAMES and not c.DATALOADER.TIE_AUGMENTATION
    assert c.SOLVER.IMS_PER_BATCH == 64 and c.DATALOADER.NUM_INSTANCE == 4
    assert c.SOLVER.MAX_EPOCHS == 60 and c.SOLVER.SEED == 1234
    assert c.SOLVER.OPTIMIZER_NAME == 'Adam' and c.SOLVER.BASE_LR == .00035
    assert c.SOLVER.PRETRAINED_LR == .000005 and c.SOLVER.WEIGHT_DECAY == .0001
    assert c.SOLVER.WEIGHT_DECAY_BIAS == .0001 and c.SOLVER.WARMUP_ITERS == 100
    assert c.M2_DIRECT_TEXT.TEMPERATURE == .07 and c.M2_DIRECT_TEXT.WEIGHT == .5
    assert c.M2_DIRECT_TEXT.NUM_CLASSES == 500
    assert c.TEST.METRIC == 'sysu' and c.TEST.FEAT_NORM == 'yes'
    assert not c.TEST.RE_RANKING and c.TEST.TOP_K_EVAL == 0

def load_text(c, train, device, num_classes):
    from tools.m2.artifacts import file_digest
    validate_config(c)
    bank = torch.load(c.M2_DIRECT_TEXT.TEXT_FEATURES_PATH, map_location='cpu', weights_only=False)
    meta, text = bank['metadata'], bank['text_bank'].detach().float()
    assert meta['implementation'] == 'm2-5-audit-v2' and meta['stage'] == 'prompt'
    assert meta['optimizer_updates'] == 10000 and meta['temperature'] == .07
    assert text.shape == (num_classes, 512) and num_classes == 500 and torch.isfinite(text).all()
    torch.testing.assert_close(text.norm(dim=1), torch.ones(num_classes), atol=1e-5, rtol=1e-4)
    # Compare to the actual loader's internal labels, not shape alone.
    files, pid_map = [], {}
    for mod in c.DATASETS.MODALITIES:
        for path, label, camera, modality in train[mod]:
            raw = int(re.search(r'(\d+)_c', Path(path).name).group(1))
            if raw in pid_map:
                assert pid_map[raw] == label
            pid_map[raw] = label
            files.append(mod + '/' + Path(path).name)
    source = meta['source']
    assert source['split'] == 'train' and source['pid_map'] == pid_map
    assert source['filenames'] == files and len(files) == 92133
    assert sorted(pid_map.values()) == list(range(500))
    actual_text = file_digest(c.M2_DIRECT_TEXT.TEXT_FEATURES_PATH)
    actual_clip = file_digest(c.MODEL.PRETRAIN_PATH)
    assert actual_text == TEXT_SHA and actual_clip == CLIP_SHA == source['clip_sha256']
    provenance = dict(text_sha256=actual_text, clip_sha256=actual_clip, prompt_updates_reused=10000,
        new_prompt_updates=0, teacher_updates=0, train_images=len(files), pid_map=pid_map,
        text_path=c.M2_DIRECT_TEXT.TEXT_FEATURES_PATH, text_shape=[500,512], projection_shape=[768,512])
    # Reuse the same already normalized table S5 used; no text tower/tokenizer.
    return text.to(device), provenance

def text_ce(features, projection, text, target, temperature=.07):
    with torch.cuda.amp.autocast(enabled=False):
        z = F.normalize(features.float() @ projection.detach().float(), dim=-1)
        logits = (z @ text.detach().float().t()) / float(temperature)
        return F.cross_entropy(logits, target.long()), logits

def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()

def config_diff(c):
    parent = CfgNode.load_cfg((Path(__file__).resolve().parents[1] / 'configs/m2_dt/m5_student_actual.yml').read_text(encoding='utf-8'))
    def flat(node, prefix=''):
        result = {}
        for key,value in node.items():
            if isinstance(value,CfgNode): result.update(flat(value,prefix+key+'.'))
            else: result[prefix+key] = list(value) if isinstance(value,tuple) else value
        return result
    a,b=flat(parent),flat(c)
    return {key:dict(parent=a.get(key),current=b.get(key)) for key in sorted(set(a)|set(b)) if a.get(key)!=b.get(key)}

def start(c, model, train, logger):
    from tools.m2.artifacts import code_commit
    text, resources = load_text(c, train, next(model.parameters()).device, model.num_classes)
    assert not model.base.clip_proj.requires_grad and model.base.clip_proj.shape == (768,512)
    out=Path(c.OUTPUT_DIR); out.mkdir(parents=True,exist_ok=True)
    if (out/'initial_state.json').exists() and not c.M2_DIRECT_TEXT.RESUME:
        raise ValueError('DT output already has initialization evidence; use a new OUTPUT_DIR or explicit RESUME')
    meta=dict(implementation=IMPLEMENTATION,variant='direct_identity_text_c0',parent_commit=PARENT,
        config=c.dump(),resources=resources,temperature=.07,weight=.5,student=dict(
            trajectory_variant='dense',acceleration_mix=0.,gain_shape=list(model.base.token_trajectory.gain.shape)))
    model.dt_metadata=meta
    (out/'effective_config.yml').write_text(c.dump(),encoding='utf-8')
    write_json(out/'config_diff_vs_m5_student.json',config_diff(c))
    if not c.M2_DIRECT_TEXT.RESUME:
        write_json(out/'initial_state.json',dict(code_commit=code_commit(),parent_commit=PARENT,
            initialization='original CLIP + seed1234 C0/head/BN; no trained student/teacher loaded',
            state_sha256=state_digest(model.state_dict()),resources=resources))
    (out/'source_and_resources.md').write_text(
        '# M2-5-DT 实际源码与资源\n\n'
        '父提交：`%s`；运行提交：`%s`。\n\n文本：`%s`，SHA-256 `%s`。\n\n'
        '原CLIP：`%s`，SHA-256 `%s`。\n\n500×512固定文本、768×512冻结P0；'
        '92133训练图的PID映射/文件顺序已核对。\n\n'
        'loss=原CE+Triplet+0.5×直接文本CE，tau=0.07，无Td/Td²/KL/教师；新增prompt与教师更新均为0。\n'
        % (PARENT,code_commit(),c.M2_DIRECT_TEXT.TEXT_FEATURES_PATH,resources['text_sha256'],
           c.MODEL.PRETRAIN_PATH,resources['clip_sha256']),encoding='utf-8')
    logger.info('M2-5-DT: fixed P5 500x512; tau=.07; CE weight=.5; teacher/prompt new updates=0; text_sha=%s CLIP_sha=%s',resources['text_sha256'],resources['clip_sha256'])
    return text

def save_training(c, epoch, model, optimizer, scheduler, scaler, iterations, updates, skips, seconds, peak):
    from tools.m2.artifacts import code_commit, rng_state
    out=Path(c.OUTPUT_DIR)
    meta=copy.deepcopy(model.dt_metadata)
    meta.update(code_commit=code_commit(),epoch=epoch,actual_iterations=iterations,
        optimizer_updates=updates,amp_skips=skips,stage_seconds=seconds,peak_cuda_memory_bytes=peak)
    state=model.state_dict()  # Plain C0, with no text bank/tower/teacher state.
    full=dict(state_dict=state,metadata=meta,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),rng=rng_state())
    torch.save(full,out/('transformer_%d.pth'%epoch))
    torch.save(dict(state_dict=state,metadata=meta),out/('transformer_%d_student.pth'%epoch))
    write_json(out/'training_budget.json',{k:meta[k] for k in ['epoch','actual_iterations','optimizer_updates',
        'amp_skips','stage_seconds','peak_cuda_memory_bytes']})
    return full

def load_deployment(model,path):
    obj=torch.load(path,map_location='cpu',weights_only=False); meta=obj['metadata']
    assert meta['implementation']==IMPLEMENTATION and meta['variant']=='direct_identity_text_c0'
    assert meta['student']==dict(trajectory_variant=model.base.token_trajectory_variant,
        acceleration_mix=model.trajectory_accel_mix,gain_shape=list(model.base.token_trajectory.gain.shape))
    model.load_state_dict(obj['state_dict'],strict=True)  # Includes classifier and BN.
    return obj

def restore(c,model,optimizer,scheduler,scaler):
    from tools.m2.artifacts import restore_rng
    obj=load_deployment(model,c.M2_DIRECT_TEXT.RESUME); meta=obj['metadata']
    assert meta['resources']==model.dt_metadata['resources']
    old,new=CfgNode.load_cfg(meta['config']),CfgNode.load_cfg(c.dump())
    for config in (old,new):
        config.defrost();config.OUTPUT_DIR='';config.M2_DIRECT_TEXT.RESUME=''
    assert old==new,'DT resume config changed'
    optimizer.load_state_dict(obj['optimizer']);scheduler.load_state_dict(obj['scheduler'])
    scaler.load_state_dict(obj['scaler']);restore_rng(obj['rng'])
    return meta
