"""收集 8 组 × 2 种读出，列出性能与配对增益两个候选。"""
import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_matrix import IDS, load_config, training_settings, verify


def collect(root):
    rows, missing = {}, []
    for name in IDS:
        run = root / (name + '_seed1234')
        paths = [run / 'training_summary.json', run / 'pre_bn/metrics.json', run / 'post_bn/metrics.json']
        absent = [str(p) for p in paths if not p.is_file()]
        if absent:
            missing.extend(absent)
            continue
        training, pre, post = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        assert training['complete'] and training['epochs_completed'] == 60, name + ' 未完整训练'
        assert training['experiment_id'] == name
        assert training['checkpoint'] == pre['checkpoint'] == post['checkpoint'], name + ' 两读出不是同一权重'
        assert Path(pre['checkpoint']).name == 'transformer_60.pth'
        reference = load_config(name)
        for readout, result in [('before', pre), ('after', post)]:
            assert result['experiment_id'] == name and result['neck_feat'] == readout
            assert not result['vtc'] and result['seed'] == 1234
            assert result['trajectory'] == reference.MODEL.TOKEN_TRAJECTORY
            assert result['environment'] == reference.EXPERIMENT.ENVIRONMENT
            assert result['normalization'] == reference.EXPERIMENT.NORMALIZATION
            resolved = yaml.safe_load(result['config'])
            expected = training_settings(reference)
            actual = training_settings(resolved)
            # 测试仅改变读出和 checkpoint；其余模型、输入和检索配置必须保持一致。
            actual['TEST']['NECK_FEAT'] = 'before'
            actual['TEST']['WEIGHT'] = ''
            assert actual == expected, name + ' 测试协议与训练配置不同'
            runtime = result['runtime']
            assert runtime['executable'] == reference.EXPERIMENT.PYTHON
            assert runtime['cudnn_benchmark'] and runtime['cudnn_deterministic']
            assert runtime['world_size'] == 1 and runtime['visible_devices'] == '0'
            for key in ['executable', 'python', 'torch', 'torchvision', 'timm', 'torch_cuda', 'cudnn', 'device']:
                assert training['runtime'][key] == runtime[key], name + ' 训练/测试环境不同：' + key
        rows[name] = {'pre': pre['overall'], 'post': post['overall'], 'amp_skips': training['amp_skips']}
    return rows, missing


def render(rows, missing):
    lines = ['# 八组配对实验结果', '', '主结果：epoch-60 的 pre-BN。补充结果：同一权重的 post-BN。单位为 %，增益为百分点。', '',
             '| ID | pre mAP | pre R1 | pre R5 | pre R10 | post mAP | post R1 | post R5 | post R10 | AMP skips |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in IDS:
        if name not in rows:
            lines.append('| ' + name + ' | ' + ' | '.join(['待完成'] * 9) + ' |')
            continue
        row = rows[name]
        values = [row[readout][key] for readout in ['pre', 'post'] for key in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']]
        lines.append('| {} | {} | {} |'.format(name, ' | '.join('{:.2f}'.format(x) for x in values), row['amp_skips']))
    lines += ['', '## 四组配对消融', '',
              '| 环境/norm | Baseline pre mAP | Trajectory pre mAP | Δ pre mAP | Δ pre R1 | Δ post mAP |',
              '|---|---:|---:|---:|---:|---:|']
    pairs = []
    for env in ['W', 'L']:
        for norm in ['S', 'C']:
            pair = env + '_' + norm
            b, t = pair + '_B', pair + '_T'
            if b not in rows or t not in rows:
                lines.append('| {} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |'.format(pair))
                continue
            baseline, trajectory = rows[b], rows[t]
            gain = trajectory['pre']['mAP'] - baseline['pre']['mAP']
            pairs.append({'pair': pair, 'trajectory_pre_mAP': trajectory['pre']['mAP'], 'delta_pre_mAP': gain})
            values = [baseline['pre']['mAP'], trajectory['pre']['mAP'], gain,
                      trajectory['pre']['Rank-1'] - baseline['pre']['Rank-1'],
                      trajectory['post']['mAP'] - baseline['post']['mAP']]
            lines.append('| {} | {} |'.format(pair, ' | '.join('{:+.2f}'.format(x) if i >= 2 else '{:.2f}'.format(x)
                                                                for i, x in enumerate(values))))
    candidates = {}
    if not missing:
        for label, key in [('performance', 'trajectory_pre_mAP'), ('ablation_gain', 'delta_pre_mAP')]:
            best = max(p[key] for p in pairs)
            candidates[label] = [p for p in pairs if p[key] == best]
        lines += ['', '## 并列保留的候选', '',
                  '- 性能最优：' + '、'.join(p['pair'] for p in candidates['performance']),
                  '- 消融增益最大：' + '、'.join(p['pair'] for p in candidates['ablation_gain']),
                  '- 本工具不自动确定主结果；两种候选由用户在八组完成后选择。',
                  '- 若增益非正，不能将该配对结果称为 Trajectory 改善。']
    else:
        lines += ['', '八组及其两读出尚未收齐，不进行候选排名。', '', '缺失文件：', '']
        lines += ['- ' + p for p in missing]
    lines += ['', '该矩阵仅为 seed=1234 的配置筛选。若依据测试集选择配置，应披露筛选过程；不能据此声称跨种子稳定优越或环境差异的单因素因果结论。', '']
    return '\n'.join(lines), candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/cache/wanghanzhi/XK/Trajectory/logs/matrix_1001'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify()
    rows, missing = collect(args.root)
    text, candidates = render(rows, missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(
        {'rows': rows, 'missing': missing, 'candidates': candidates}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已汇总 {} / 8 组：{}'.format(len(rows), args.output))


if __name__ == '__main__':
    main()
