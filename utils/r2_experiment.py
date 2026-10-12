"""E3 identity, exact optimizer scope and observations without training effects."""
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from yacs.config import CfgNode
from utils.m3_artifacts import file_hash

PARENT='a432df3cc59c92aab91afdacd43ea1bf81d522fd'
REPO=Path(__file__).resolve().parents[1]
CONTROL_NAMES=('context','q.weight','k.weight','v.weight','o.weight')


def code_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO),text=True).strip()


def r2_signature(c):
    if not c.R2.ENABLED:
        return None
    if (c.R2.EXPERIMENT!='m3_no_weight_decay' or c.R2.PARENT_CODE_COMMIT!=PARENT or
        c.R2.CONTROLLER_WEIGHT_DECAY!=0. or not c.M2.ENABLED or
        c.M2.VARIANT!='deep_text_c0_control'):
        raise ValueError('E3 requires original M3 and only controller weight decay=0')
    return dict(experiment=c.R2.EXPERIMENT,parent_code_commit=PARENT,controller_weight_decay=0.)


def controller_parameters(model):
    owner=model.module if hasattr(model,'module') else model
    controller=owner.base.deep_c0_controller
    params=dict(controller.named_parameters())
    if tuple(params)!=CONTROL_NAMES or sum(p.numel() for p in params.values())!=186368:
        raise ValueError('E3 expected five controller tensors, 186368 elements')
    if not all(p.requires_grad for p in params.values()):
        raise ValueError('E3 controller parameters must retain original trainability')
    return params


def config_differences(c,allow_path_mapping=False):
    before=CfgNode.load_cfg((REPO/'configs/r2_m3_parent_actual.yml').read_text(encoding='utf-8'))
    def flat(node,prefix=''):
        result={}
        for key,value in node.items():
            if isinstance(value,CfgNode):
                result.update(flat(value,prefix+key+'.'))
            else:
                result[prefix+key]=list(value) if isinstance(value,tuple) else value
        return result
    old,new=flat(before),flat(c)
    diff={key:dict(parent=old.get(key),current=value) for key,value in new.items() if old.get(key)!=value}
    allowed={'OUTPUT_DIR','M2.RESUME'}
    if allow_path_mapping:
        allowed.update(('DATASETS.ROOT_DIR','DATASETS.SUBDIR','MODEL.PRETRAIN_PATH','MODEL.TEXT_CLIP_PATH',
                        'M2.CLIP_PATH','M2.TEXT_BANK','M2.ID_PROMPT_WEIGHT','M2.DEEP_CONTEXT_INIT'))
    unexpected=[key for key in diff if key not in allowed and not key.startswith('R2.')]
    if unexpected:
        raise ValueError('E3 changed parent training settings: '+str(unexpected))
    return diff


def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((value.dtype,tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def verify_parent_artifacts(c):
    expected=json.loads((REPO/'configs/r2_m3_parent_artifacts.json').read_text(encoding='utf-8'))
    assert expected['parent_code_commit']==PARENT
    actual=dict(clip=file_hash(c.M2.CLIP_PATH),m3_bank=file_hash(c.M2.TEXT_BANK))
    if actual!=expected['sha256']:
        raise ValueError('E3 CLIP/bundled deep-slot bank differs from original verified M3')
    return actual


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def startup_manifest(c,model,optimizer,logger):
    signature=r2_signature(c)
    diff=config_differences(c,True)
    artifacts=verify_parent_artifacts(c)
    controller_parameters(model)
    names={id(p):name for name,p in model.named_parameters()}
    groups=[dict(index=i,parameters=[names[id(p)] for p in g['params']],
        numel=sum(p.numel() for p in g['params']),peak_lr=g.get('initial_lr',g['lr']),
        lr=g['lr'],weight_decay=g['weight_decay']) for i,g in enumerate(optimizer.param_groups)]
    parent=optimizer.r2_parent_groups
    changes=[]
    for old,new in zip(parent,groups):
        assert all(old[key]==new[key] for key in ('index','parameters','numel','peak_lr'))
        if old['weight_decay']!=new['weight_decay']:
            changes.append(dict(parameter=new['parameters'][0],parent=old['weight_decay'],current=new['weight_decay']))
    assert len(parent)==len(groups)
    assert {v['parameter'] for v in changes}=={'base.deep_c0_controller.'+n for n in CONTROL_NAMES}
    assert all(v['parent']==1e-4 and v['current']==0. for v in changes)
    out=Path(c.OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    if (out/'r2_initial_state.json').exists() and not c.M2.RESUME:
        raise ValueError('E3 output already contains initial manifest; resume same full state or choose a new directory')
    if not c.M2.RESUME:
        initial_hash=tensor_digest(model.state_dict())
        model.m2_metadata['r2_initial_state_sha256']=initial_hash
        write_json(out/'r2_initial_state.json',dict(signature=signature,code_commit=code_commit(),
            student_initial_state_sha256=initial_hash,clip_path=c.M2.CLIP_PATH,clip_sha256=artifacts['clip'],
            bank_path=c.M2.TEXT_BANK,bank_sha256=artifacts['m3_bank'],new_prompt_updates=0,
            conditional_teacher='not_applicable',planned_epochs=60))
        write_json(out/'optimizer_groups.json',dict(current=groups,parent=parent,changes=changes,defaults=optimizer.defaults))
        write_json(out/'config_diff_vs_parent.json',diff)
        (out/'code_commit.txt').write_text(code_commit()+'\n',encoding='utf-8')
        (out/'git_diff.patch').write_bytes(subprocess.check_output(['git','diff',PARENT,'HEAD'],cwd=str(REPO)))
    if not (out/'config_merged.yml').exists():
        (out/'config_merged.yml').write_text(c.dump(),encoding='utf-8')
    logger.info('E3复用deep-slot bank=%s sha256=%s；CLIP=%s sha256=%s；新增prompt优化步0，无M5条件教师；学生计划60epoch；resume=%s',
                c.M2.TEXT_BANK,artifacts['m3_bank'],c.M2.CLIP_PATH,artifacts['clip'],c.M2.RESUME or '无，从原初始化开始')


def parameter_snapshot(model):
    return {name:dict(norm_fp64=p.detach().double().norm().item(),
                     max_abs_fp64=p.detach().double().abs().max().item())
            for name,p in controller_parameters(model).items()}


def gradient_snapshot(model):
    result={}
    for name,p in controller_parameters(model).items():
        if p.grad is None:
            result[name]=dict(status='None',norm_fp64=None,max_abs_fp64=None)
        elif not torch.isfinite(p.grad).all():
            result[name]=dict(status='nonfinite',norm_fp64=None,max_abs_fp64=None)
        else:
            maximum=p.grad.detach().double().abs().max().item()
            result[name]=dict(status='all_zero' if maximum==0 else 'finite_nonzero',
                norm_fp64=p.grad.detach().double().norm().item(),max_abs_fp64=maximum)
    return result


def total_gradient_norm(model):
    return torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm().item()
