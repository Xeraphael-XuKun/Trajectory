"""Historical C0/M5 strict visual reload for matched query comparisons; no retraining."""
import ast
import re
from pathlib import Path
import yaml
from yacs.config import CfgNode
from config import cfg as defaults
from tools.m2.artifacts import file_digest, VERSION
from tools.iclr_exp.neutral_assets import TEACHER_COMMIT, json_digest


def saved_config(text, logger_format=False):
    c=CfgNode.load_cfg(text)
    literal=yaml.load(text,Loader=yaml.BaseLoader) if logger_format else None
    def complete(current,default,raw):
        for key,value in default.items():
            if key not in current:
                current[key]=value.clone() if isinstance(value,CfgNode) else value
            elif isinstance(value,CfgNode):
                complete(current[key],value,raw.get(key,{}) if raw else None)
            elif raw and isinstance(value,str) and not isinstance(current[key],str):
                current[key]=raw[key]
    complete(c,defaults,literal)
    if isinstance(c.DATASETS.MODALITIES,str):c.DATASETS.MODALITIES=ast.literal_eval(c.DATASETS.MODALITIES)
    return c


def reference_checkpoint(method,obj,resource_config,checkpoint,train_log=None):
    if method=='m5':
        meta=obj['metadata']
        if (meta['implementation']!=VERSION or meta['variant']!='conditional_teacher_distill' or
            meta['stage']!='student' or meta['epoch']!=60 or meta['code_commit']!=TEACHER_COMMIT or
            meta['student']!=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=[11,1,129,768])):
            raise ValueError('not the selected historical M5 epoch60 student')
        if meta['teacher_sha256']!=file_digest(resource_config.ICLR_EXP.ASSETS.TEACHER_CKPT) or meta['text_sha256']!=file_digest(resource_config.ICLR_EXP.ASSETS.TEXT_BANK):
            raise ValueError('historical M5 links different teacher/text resources')
        c=saved_config(meta['config'])
        if meta['source']['clip_sha256']!=file_digest(resource_config.ICLR_EXP.ASSETS.ORIGINAL_CLIP):
            raise ValueError('historical M5 original CLIP differs')
        evidence=dict(historical_metadata={k:meta[k] for k in ('implementation','variant','stage','epoch',
            'base_commit','code_commit','text_sha256','teacher_sha256','teacher_architecture','student','config')},
            source=dict(split=meta['source']['split'],pid_mapping=meta['source']['pid_map'],
                train_manifest_sha256=json_digest(meta['source']['filenames']),
                num_train_images=len(meta['source']['filenames']),condition_mask=meta['source']['condition_mask'].tolist()),
            teacher_and_text_links_verified=True)
    else:
        if not train_log or Path(checkpoint).name!='transformer_60.pth':
            raise ValueError('historical C0 needs its original train log and transformer_60.pth')
        log=Path(train_log).read_text(encoding='utf-8')
        text=re.split(r'\n\d{4}-\d\d-\d\d ',log.split('Running with config:\n',1)[1],1)[0]
        if 'Epoch[60]' not in log:raise ValueError('C0 log has no epoch60 training evidence')
        c=saved_config(text,logger_format=True)
        evidence=dict(train_log=str(train_log),train_log_sha256=file_digest(train_log),historical_config=text,
            checkpoint_epoch_evidence='original log reaches Epoch[60]; fixed transformer_60.pth',
            caveat='legacy checkpoint has no embedded epoch/source manifest; provenance linked through its original train log')
    if (not c.MODEL.TOKEN_TRAJECTORY or c.MODEL.TOKEN_TRAJECTORY_VARIANT!='dense' or c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX!=0 or
        c.MODEL.TRANSFORMER_TYPE!='vit_base_clip' or c.MODEL.PE_TYPE!='learnable' or c.MODEL.PE_LAYERWISE!='none' or
        any((c.MODEL.TEXT_ALIGN,c.MODEL.VPR,c.MODEL.MOD_DELTA,c.MODEL.SIE_CAMERA,c.MODEL.SIE_VIEW,c.MODEL.CE_SPLIT_MODALITY,c.MODEL.CE_SPLIT_VIEW,c.MODEL.LAYER_SCALE,c.MODEL.CHART_INPUT_NORM))):
        raise ValueError('reference is not the original dense velocity-only C0 visual architecture')
    if method=='c0' and c.M2.ENABLED:raise ValueError('C0 historical config enables M2')
    for parent,key in (('INPUT','SIZE_TEST'),('INPUT','SIZE_TRAIN'),('INPUT','PIXEL_MEAN'),('INPUT','PIXEL_STD'),('MODEL','STRIDE_SIZE'),('DATASETS','MODALITIES'),('DATASETS','AERIAL_CAMS')):
        if list(getattr(getattr(c,parent),key))!=list(getattr(getattr(resource_config,parent),key)):
            raise ValueError('reference preprocessing/mapping mismatch: '+parent+'.'+key)
    c.ICLR_EXP.ENABLED=False;c.ICLR_EXP.ASSETS.ORIGINAL_CLIP=resource_config.ICLR_EXP.ASSETS.ORIGINAL_CLIP;c.M2.ENABLED=False
    # Full state is strictly restored; no dependency on locally resized initialization.
    c.MODEL.PRETRAIN_CHOICE='no'
    c.TEST.NECK_FEAT='after';c.TEST.FEAT_NORM='yes';c.TEST.METRIC='sysu';c.TEST.RE_RANKING=False;c.TEST.TOP_K_EVAL=0
    c.DATASETS.ROOT_DIR=resource_config.DATASETS.ROOT_DIR;c.DATASETS.SUBDIR=resource_config.DATASETS.SUBDIR;c.DATASETS.PROTOCOL='ALL'
    state=obj.get('state_dict',obj)
    meta=dict(implementation='historical-'+method+'-strict-v1',variant=method,epoch=60,reference_evidence=evidence,
        original_clip_sha256=file_digest(resource_config.ICLR_EXP.ASSETS.ORIGINAL_CLIP),
        student=dict(trajectory_variant='dense',acceleration_mix=0.,gain_shape=[11,1,129,768]))
    return c,meta,state
