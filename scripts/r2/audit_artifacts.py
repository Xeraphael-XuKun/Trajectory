"""W0: explicit paths, actual metadata, original validators, strict model loads."""
import argparse
import gc
import subprocess
from pathlib import Path

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--formal', action='store_true')
    args = ap.parse_args()
    paths, out = resources(args.resources), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(environment=environment(), paths=paths, artifacts={}, checks={},
                    source_parent=paths['m5_parent'],
                    device=args.device, teacher_optimization_steps=0, prompt_optimization_steps=0)
    try:
        execution_mode(args.formal)
        manifest['formal_original_environment'] = args.formal
        manifest['code_commit'] = code_commit()
        for role in ('clip', 'm5_teacher', 'm5_text', 'm5_prompt', 'm5_student', 'm3_bank', 'm3_student', 'c0_student'):
            p = Path(paths[role])
            uses = {'clip':['W0','D1'], 'm5_teacher':['W0','D1','E1','E2'],
                    'm5_text':['W0','D1','E1','E2'], 'm5_prompt':['W0','D1','E1','E2'],
                    'm5_student':['W0','D1','D2'], 'm3_bank':['W0','E3'],
                    'm3_student':['W0','D3'], 'c0_student':['W0','D2']}
            entry = dict(path=str(p.resolve()), exists=p.is_file(), bytes=p.stat().st_size if p.is_file() else None,
                         used_by=uses[role])
            manifest['artifacts'][role] = entry
            if not p.is_file():
                raise FileNotFoundError(str(p))
            entry['sha256'] = digest(p)
            if role != 'clip':
                obj = load(p)
                entry['top_keys'] = list(obj)
                meta = obj.get('metadata', {})
                entry['metadata'] = {k: v for k, v in meta.items() if k in
                    ('epoch', 'stage', 'implementation', 'variant', 'code_commit', 'base_commit',
                     'actual_iterations', 'optimizer_updates', 'amp_skips', 'text_sha256', 'teacher_sha256', 'prompt_sha256')}
                state = obj.get('state_dict')
                if state:
                    entry['state_tensors'] = len(state)
                    entry['nonfinite_keys'] = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
                    assert not entry['nonfinite_keys']
                del obj
                gc.collect()
            print('RESOURCE', role, entry['bytes'], flush=True)
        c = m5_config(paths)
        from yacs.config import CfgNode
        actual = CfgNode.load_cfg(load(paths['m5_student'])['metadata']['config'])
        disk = CfgNode.load_cfg(Path(paths['m5_config']).read_text(encoding='utf-8'))
        assert actual == disk
        manifest['checks']['m5_merged_equals_checkpoint_config'] = True
        assert (c.M2.KD_WEIGHT, c.M2.KD_TEMPERATURE, c.M2.TEXT_TEMPERATURE) == (.5, 2., .07)
        (out/'config_m5_local.yml').write_text(c.dump(), encoding='utf-8')
        (out/'config_m5_historical.yml').write_text(actual.dump(), encoding='utf-8')
        (out/'config_c0_local.yml').write_text(c0_config(paths).dump(), encoding='utf-8')
        (out/'config_c0_historical.txt').write_text(c0_historical_config(paths), encoding='utf-8')
        def flatten(config, prefix=''):
            result={}
            for key,value in config.items():
                name=prefix+key
                if isinstance(value,CfgNode):
                    result.update(flatten(value,name+'.'))
                else:
                    result[name]=value
            return result
        before,after=flatten(actual),flatten(c)
        write_json(out/'config_diff_vs_parent.json',{
            key:dict(historical=before.get(key),local=after.get(key),reason='explicit path mapping only')
            for key in after if before.get(key)!=after.get(key)})
        from tools.m2.artifacts import validate_config, validate_text_bank
        validate_config(c)
        source = validate_text_bank(load(paths['m5_text']), c, 500)
        train, query, gallery = (index_split(paths, s) for s in ('train', 'query', 'test'))
        assert len(train) == len(source['filenames']) == 92133
        assert len(query) == 6405 and len(gallery) == 93609
        assert not {r['raw_pid'] for r in train} & {r['raw_pid'] for r in query+gallery}
        manifest['checks']['train_source_pid_order_clip_prompt_text'] = True
        manifest['data_counts'] = dict(train=len(train), query=len(query), gallery=len(gallery),
            train_ids=len(source['pid_map']), query_aerial=sum(r['platform'] for r in query),
            gallery_aerial=sum(r['platform'] for r in gallery))
        # Original M3 source/signature check in its own process, avoiding module mixing.
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('audit_m3.py')),
                                 '--resources', args.resources, '--output', str(out/'m3_checks.json')],
                                capture_output=True, text=True)
        (out/'m3_audit_output.txt').write_text(result.stdout+result.stderr, encoding='utf-8')
        assert result.returncode == 0, result.stderr
        manifest['checks']['m3_original_source_and_bank'] = json.loads((out/'m3_checks.json').read_text(encoding='utf-8'))
        for role in ('teacher', 'm5', 'c0'):
            model = teacher(paths, args.device)[0] if role == 'teacher' else student(paths, args.device, role)[0]
            manifest['checks'][role+'_strict_load'] = True
            if role == 'm5':
                from solver.make_optimizer import make_optimizer
                optimizer,_ = make_optimizer(c,model,torch.nn.Linear(1,1))
                names={id(p):n for n,p in model.named_parameters()}
                write_json(out/'optimizer_groups_m5_parent.json',[
                    dict(index=i,parameters=[names[id(p)] for p in g['params']],
                         numel=sum(p.numel() for p in g['params']),lr=g['lr'],weight_decay=g['weight_decay'])
                    for i,g in enumerate(optimizer.param_groups)])
            if role == 'teacher':
                assert not model.training and all(not p.requires_grad for p in model.parameters())
                manifest['checks']['teacher_frozen_eval'] = True
                manifest['checks']['teacher_diagnostic_compatibility'] = getattr(model, 'diagnostic_compatibility', {})
            del model
            gc.collect()
            print('STRICT_LOAD', role, 'PASS', flush=True)
        manifest['status'] = 'COMPLETED' if args.formal else 'COMPLETED_WITH_CAVEAT'
        manifest['caveats'] = ['本核验严格重载权重；全数据GPU推理另由D1/D2确认。']
        if not args.formal:
            manifest['caveats'].append('本地环境不同于历史A800环境；未检查云端当前可用性。')
    except Exception as exc:
        manifest['status'] = 'FAILED'
        manifest['error'] = repr(exc)
        raise
    finally:
        write_json(out/'artifact_manifest.json', manifest)


if __name__ == '__main__':
    main()
