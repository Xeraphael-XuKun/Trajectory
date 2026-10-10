"""提取HF四组、历史HI与C0检索和训练诊断；不改变原始文件。"""
import argparse
import ast
import json
import re
import statistics
from pathlib import Path

import yaml


def read(path):
    return path.read_text(encoding='utf-8-sig')


def lines(path):
    return [json.loads(x) for x in read(path).splitlines() if x.strip()]


def flat(value, prefix=''):
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result.update(flat(item, prefix + str(key) + '.'))
        else:
            result[prefix + str(key)] = item
    return result


def results(text):
    sections = text.split('Validation Results')[1:]
    return [[float(re.search(pattern, section).group(1)) for pattern in
             [r'INFO: mAP:\s*([\d.]+)%', r'CMC curve, Rank-1\s*:\s*([\d.]+)%',
              r'CMC curve, Rank-5\s*:\s*([\d.]+)%', r'CMC curve, Rank-10\s*:\s*([\d.]+)%']]
            for section in sections]


def evaluation(directory):
    data = {}
    for neck in ['post', 'pre']:
        path = next((directory / ('eval_' + neck + 'BN')).glob('test_log_*.txt'))
        text = read(path)
        pairs = {}
        views = {}
        for kind, a, b, *values in re.findall(
                r'(Pair|View) \(Q:(\w+), G:(\w+)\) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-5: ([\d.]+)%, Rank-10: ([\d.]+)%, mINP: ([\d.]+)%', text):
            (pairs if kind == 'Pair' else views)[a + '-' + b] = list(map(float, values))
        data[neck] = {'path': str(path), 'metrics': results(text)[-1], 'pairs': pairs, 'views': views,
                      'weight': re.findall(r'^  WEIGHT: (.*)$', text, re.M)[-1].strip(),
                      'neck': re.findall(r'^  NECK_FEAT: (.*)$', text, re.M)[-1].strip(),
                      'view_counts': re.findall(r'Starting view-pair evaluation .*', text),
                      'runtime': re.findall(r'Runtime: (.*)', text)}
    return data


def extract(root):
    data = {'groups': {}, 'references': {}}
    names = ['HD2_HF1', 'HD3_HF1', 'HD3_HF2', 'HD3_HF3']
    for name in names:
        directory = root / 'history_followup_1009' / (name + '_seed1234')
        log_path = next(directory.glob('train_log_*.txt'))
        text = read(log_path)
        console = read(directory / 'console.log')
        epochs = lines(directory / 'history_epoch.jsonl')
        gradients = lines(directory / 'history_gradients.jsonl')
        config = yaml.safe_load(read(directory / 'resolved_config.yml'))
        selected = yaml.safe_load(read(directory / 'selected_config.yml'))
        evaluation_data = evaluation(directory)
        logged_results = results(text)
        times = [float(x) for x in re.findall(r'Time per batch: ([\d.]+)\[s\]', text)]
        # yacs literal decoding and explicit launcher overrides are expected.
        def decoded(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            return value
        selected_differences = {k: [v, flat(config)[k]] for k, v in flat(selected).items()
                                if k not in ['OUTPUT_DIR', 'MODEL.PRETRAIN_PATH', 'DATASETS.ROOT_DIR']
                                and decoded(v) != flat(config)[k]}
        checks = {
            'epochs_1_to_60': [e['epoch'] for e in epochs] == list(range(1, 61)),
            'done_1_to_60': list(map(int, re.findall(r'Epoch (\d+) done\.', text))) == list(range(1, 61)),
            'inline_equal': [json.loads(x) for x in re.findall(r'History epoch summary: (\{.*\})', text)] == epochs,
            'full_horizon': config['SOLVER']['MAX_EPOCHS'] == 60 and config['HISTORY']['STOP_AFTER_EPOCH'] == 0,
            'gradient_samples_complete': [(g['epoch'], g['iteration']) for g in gradients] ==
                [(e['epoch'], j) for e in epochs for j in range(1, e['iterations'] + 1) if j == 1 or j % 50 == 0],
            'final_train_post_equal': logged_results[-1] == evaluation_data['post']['metrics'],
            'same_checkpoint': evaluation_data['post']['weight'] == evaluation_data['pre']['weight'] and
                evaluation_data['post']['weight'].endswith(name + '_seed1234/transformer_60.pth'),
            'readouts_correct': evaluation_data['post']['neck'] == 'after' and evaluation_data['pre']['neck'] == 'before',
            'train_console_equal': results(console) == logged_results + [evaluation_data['post']['metrics'], evaluation_data['pre']['metrics']],
            'selected_config_after_decode_and_overrides': not selected_differences,
            'six_periodic_evaluations': len(logged_results) == 6,
            'loss_and_diagnostics_finite': all(not isinstance(v, float) or (-float('inf') < v < float('inf'))
                                              for e in epochs for v in flat(e).values()),
            'baseline_and_old_modules_disabled': not any(config['MODEL'][k] for k in
                ['TOKEN_TRAJECTORY', 'TEXT_ALIGN', 'MOD_DELTA', 'VPR', 'JPM', 'SIE_CAMERA', 'SIE_VIEW']),
        }
        gradient_summary = {}
        for e in epochs:
            good = [g for g in gradients if g['epoch'] == e['epoch'] and not g['amp_skipped']]
            def median(values):
                values = [x for x in values if x is not None]
                return statistics.median(values) if values else None
            families = e['optimization']['parameters']
            gradient_summary[e['epoch']] = {
                'valid': len(good),
                'global_grad': median(g['global_grad_preclip_norm'] for g in good),
                'aux_to_identity': median(g['prediction_weighted_grad_norm']['predictors'] / g['identity_grad_norm']['predictors']
                                          for g in good if g['identity_grad_norm']['predictors']),
                'decay_to_data': {f: median(g['decay_to_data_norm_ratio'][f] for g in good) for f in families},
                'identity_norm': {f: median(g['identity_grad_norm'][f] for g in good) for f in families},
            }
        record = {'directory': str(directory), 'train_path': str(log_path), 'source': read(directory / 'source_commit.txt').strip(),
                  'status': read(directory / 'source_status.txt').strip(), 'config': config, 'epochs': epochs,
                  'initial': json.loads(read(directory / 'history_optimizer_initial.json')),
                  'evaluations': evaluation_data, 'periodic': dict(zip(range(10, 61, 10), logged_results)), 'checks': checks,
                  'selected_unexpected_differences': selected_differences,
                  'runtime': re.findall(r'Runtime: (.*)', text), 'iterations': sum(e['iterations'] for e in epochs),
                  'amp_skips': sum(e['amp_skipped_steps'] for e in epochs), 'gradient_records': len(gradients),
                  'sampled_skips': sum(g['amp_skipped'] for g in gradients), 'gradient_summary': gradient_summary,
                  'training_minutes_est': sum(e['iterations'] * t for e, t in zip(epochs, times)) / 60,
                  'loaded': re.findall(r'loaded \d+ / \d+ tensors[^\n]*', console),
                  'dataset_counts': re.findall(r'^\s*(train|query|gallery)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*$', console, re.M),
                  'fatal_lines': [x for x in console.splitlines() if re.search(r'Traceback|OutOfMemoryError|\bLoss: (nan|inf)\b', x, re.I)],
                  'local_weights': [str(x) for x in directory.glob('*.pth')]}
        if name in ['HD2_HF1', 'HD3_HF1']:
            short = lines(root / 'history_optimization_1009' / (name[:3] + '_e5_seed1234') / 'history_epoch.jsonl')
            record['first5_match_HD'] = short == epochs[:5]
        data['groups'][name] = record
        print(name, evaluation_data['post']['metrics'], evaluation_data['pre']['metrics'], checks)
        print('steps/skips', record['iterations'], record['amp_skips'], 'source', record['source'], 'load', record['loaded'], 'HD exact', record.get('first5_match_HD'))
        print('curves', record['periodic'])
        final = epochs[-1]
        print('e60 params', final['optimization']['parameters'])
        print('e60 ratios', [round(v['correction_ratio'], 5) for v in final['layers'].values()], 'cls', [round(v.get('cls_correction_ratio', 0), 5) for v in final['layers'].values()])
        print('e60 pred gains', [round(v.get('prediction_gain_ratio_of_means', 0), 5) for v in final['layers'].values()])
        print('e60 grad', gradient_summary[60])
    for name in ['C0', 'T1', 'T2']:
        data['references'][name] = evaluation(root / 'trajectory_extensions_1002' / (name + '_seed1234'))
    for name in ['HI0', 'HI1', 'HI2', 'HI3']:
        data['references'][name] = evaluation(root / 'history_innovation_1009' / (name + '_seed1234'))
        data['references'][name]['epochs'] = lines(root / 'history_innovation_1009' / (name + '_seed1234') / 'history_epoch.jsonl')
    data['references']['baseline'] = {'post': {'metrics': [12.38, 31.02, 47.76, 55.80]},
                                      'pre': {'metrics': [11.99, 31.10, 48.57, 56.69]}, 'source': '用户给定历史结果'}
    reference = flat(data['groups']['HD3_HF1']['config'])
    for name, record in data['groups'].items():
        record['config_diff_from_HD3_HF1'] = {k: v for k, v in flat(record['config']).items() if reference[k] != v}
        print(name, 'diff', record['config_diff_from_HD3_HF1'])
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', default='E:/CVPR2027/ref/Trajectory_runs')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    data = extract(Path(args.runs_root))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'parsed_hf.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
