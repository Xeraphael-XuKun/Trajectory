"""E2 identity, actual-parent config comparison and startup provenance."""
import hashlib
import json
import subprocess
from pathlib import Path
from yacs.config import CfgNode
from tools.m2.artifacts import code_commit, file_digest

PARENT = 'db8a5c65ba0b7b5f0c62d00d2adb2f5e3bfe1177'
REPO = Path(__file__).resolve().parents[1]


def r2_signature(c):
    if not c.R2.ENABLED:
        return None
    if (not c.M2.ENABLED or c.M2.TRAIN_STAGE!='student' or
        c.R2.EXPERIMENT!='m5_one_sided' or c.R2.KD_MODE!='one_sided_target' or
        c.R2.PARENT_CODE_COMMIT!=PARENT):
        raise ValueError('E2 requires original M5 student and the isolated one_sided_target mode')
    return dict(experiment=c.R2.EXPERIMENT,kd_mode=c.R2.KD_MODE,parent_code_commit=PARENT)


def actual_parent():
    return CfgNode.load_cfg((REPO/'configs/r2_m5_parent_actual.yml').read_text(encoding='utf-8'))


def config_differences(c, allow_path_mapping=False):
    def flat(node,prefix=''):
        result={}
        for key,value in node.items():
            name=prefix+key
            if isinstance(value,CfgNode):
                result.update(flat(value,name+'.'))
            else:
                # YAML dumps tuple-valued YACS settings as lists; merge restores
                # the original default tuple type. Compare sequence contents.
                result[name]=list(value) if isinstance(value,tuple) else value
        return result
    before,after=flat(actual_parent()),flat(c)
    diff={key:dict(parent=before.get(key),current=value) for key,value in after.items()
          if before.get(key)!=value}
    allowed={'OUTPUT_DIR','M2.RESUME'}
    if allow_path_mapping:
        allowed.update(('DATASETS.ROOT_DIR','DATASETS.SUBDIR','MODEL.PRETRAIN_PATH','M2.CLIP_PATH',
                        'M2.PROMPT_WEIGHT','M2.TEXT_BANK','M2.TEACHER_WEIGHT'))
    unexpected=[key for key in diff if key not in allowed and not key.startswith('R2.')]
    if unexpected:
        raise ValueError('E2 changed parent settings: '+str(unexpected))
    return diff


def tensor_digest(state):
    """One startup fingerprint for exact initialization, not per-step hashing."""
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.dtype,tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def verify_parent_artifacts(c):
    expected=json.loads((REPO/'configs/r2_m5_parent_artifacts.json').read_text(encoding='utf-8'))
    assert expected['parent_code_commit']==PARENT
    paths=dict(clip=c.M2.CLIP_PATH,m5_teacher=c.M2.TEACHER_WEIGHT,
               m5_text=c.M2.TEXT_BANK,m5_prompt=c.M2.PROMPT_WEIGHT)
    actual={role:file_digest(path) for role,path in paths.items()}
    if actual!=expected['sha256']:
        raise ValueError('E2 resources differ from the verified historical M5 artifacts')
    if Path(c.MODEL.PRETRAIN_PATH).resolve()!=Path(c.M2.CLIP_PATH).resolve():
        if file_digest(c.MODEL.PRETRAIN_PATH)!=actual['clip']:
            raise ValueError('E2 student initialization is not the original verified CLIP')
    return actual


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def startup_manifest(c,model,optimizer,logger):
    signature=r2_signature(c)
    if signature is None:
        return
    diff=config_differences(c,allow_path_mapping=True)
    artifacts=verify_parent_artifacts(c)
    out=Path(c.OUTPUT_DIR)
    out.mkdir(parents=True,exist_ok=True)
    initial=out/'r2_initial_state.json'
    if initial.exists() and not c.M2.RESUME:
        raise ValueError('E2 output already contains an initial manifest; use same-experiment M2.RESUME or a new output directory')
    names={id(p):name for name,p in model.named_parameters()}
    if not c.M2.RESUME:
        groups=[dict(index=i,parameters=[names[id(p)] for p in group['params']],
            numel=sum(p.numel() for p in group['params']),lr=group['lr'],
            peak_lr=group.get('initial_lr',group['lr']),weight_decay=group['weight_decay'])
            for i,group in enumerate(optimizer.param_groups)]
        state_hash=tensor_digest(model.state_dict())
        model.m5_metadata['r2_initial_state_sha256']=state_hash
        write_json(initial,dict(signature=signature,code_commit=code_commit(),
            student_initial_state_sha256=state_hash,initialization='original CLIP + same seeded C0/head/BN initialization',
            teacher_sha256=artifacts['m5_teacher'],text_sha256=artifacts['m5_text'],
            prompt_sha256=artifacts['m5_prompt'],clip_sha256=artifacts['clip'],
            teacher_optimization_steps_planned=0,prompt_optimization_steps_planned=0,student_epochs=60))
        write_json(out/'optimizer_groups.json',groups)
        write_json(out/'config_diff_vs_parent.json',diff)
        (out/'code_commit.txt').write_text(code_commit()+'\n',encoding='utf-8')
        patch=subprocess.check_output(['git','diff',PARENT,'HEAD'],cwd=str(REPO))
        (out/'git_diff.patch').write_bytes(patch)
    logger.info('E2复用教师=%s sha256=%s；文本=%s sha256=%s；prompt优化步=0，教师优化步=0，学生计划60epoch；resume=%s',
        c.M2.TEACHER_WEIGHT,model.m5_metadata['teacher_sha256'],c.M2.TEXT_BANK,
        model.m5_metadata['text_sha256'],c.M2.RESUME or '无，从原始CLIP开始')
