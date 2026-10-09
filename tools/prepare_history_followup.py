"""诊断完成后显式选择优化配置，生成HF1-HF3；不替用户自动选优。"""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(choice, output_root):
    base = yaml.safe_load((ROOT / 'configs' / ('history_' + choice + '.yml')).read_text(encoding='utf-8'))
    destination = Path(output_root) / choice
    destination.mkdir(parents=True, exist_ok=False)
    for name, scope, weight in [('HF1', 'patch', 1.), ('HF2', 'patch', 0.), ('HF3', 'all', 1.)]:
        c = yaml.safe_load(yaml.safe_dump(base))
        c['HISTORY'].update(STOP_AFTER_EPOCH=0, TOKEN_SCOPE=scope, LOSS_WEIGHT=weight)
        c['SOLVER'].update(MAX_EPOCHS=60, EVAL_PERIOD=10, CHECKPOINT_PERIOD=60)
        c['OUTPUT_DIR'] = './logs/history_followup_1009/' + choice + '_' + name + '_seed1234'
        path = destination / ('history_' + name + '.yml')
        path.write_text('# 正式60epoch实验；优化配置明确选自' + choice + '。\n' +
                        yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding='utf-8')
        print(path)
    (destination / 'selection.txt').write_text('优化设置选择：' + choice + '\n不是程序自动选优。\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--choice', required=True, choices=['HD0', 'HD1', 'HD2', 'HD3'])
    p.add_argument('--output-root', default=str(ROOT / 'configs/history_followup'))
    args = p.parse_args()
    prepare(args.choice, args.output_root)
