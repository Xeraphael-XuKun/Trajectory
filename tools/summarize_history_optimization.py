"""汇总HD诊断，不按残差大小自动挑选配置。"""
import argparse
import json
from pathlib import Path

def fmt(x):
    return '不可用' if x is None else '{:.3e}'.format(x)

def main(args):
    rows = []; details = []
    for directory in sorted(Path(args.runs_root).glob('HD*_e*_seed1234*')):
        source = directory / 'history_epoch.jsonl'
        if not source.exists(): continue
        epochs = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        gradients_file = directory / 'history_gradients.jsonl'
        gradients = [json.loads(line) for line in gradients_file.read_text(encoding='utf-8').splitlines()] if gradients_file.exists() else []
        config = json.loads((directory / 'history_optimizer_initial.json').read_text(encoding='utf-8'))
        end = epochs[-1]; ps = end['optimization']['parameters']
        rows.append([directory.name, str(end['epoch']), str(config['schedule_epochs']),
                     fmt(ps['gain']['effective_rms']), fmt(ps['anchors']['rms']), fmt(ps['correctors']['rms']),
                     fmt(max(v['correction_ratio'] for v in end['layers'].values())),
                     str(sum(e['amp_skipped_steps'] for e in epochs)), str(len(gradients)),
                     '是' if (directory/'history_stop.json').exists() else '否'])
        details.append('\n## ' + directory.name + '\n\n| epoch | 有效gain RMS | Anchor RMS | Corrector RMS | 最大层均修正比例 | CE | Triplet |\n| --- | --- | --- | --- | --- | --- | --- |\n')
        for e in epochs:
            p = e['optimization']['parameters']
            values = [e['epoch'],fmt(p['gain']['effective_rms']),fmt(p['anchors']['rms']),fmt(p['correctors']['rms']),
                      fmt(max(v['correction_ratio'] for v in e['layers'].values())),fmt(e['ce']),fmt(e['triplet'])]
            details.append('| '+' | '.join(map(str, values))+' |\n')
        valid = [g for g in gradients if not g['amp_skipped']]
        if valid:
            last = valid[-1]
            details.append('\n最后一个未跳步的诊断batch（不代表epoch均值）：\n\n| 部分 | 身份梯度范数 | 原始预测梯度范数 | clip后总数据梯度 | 衰减项 | 衰减/数据 |\n| --- | --- | --- | --- | --- | --- |\n')
            for family in last['identity_grad_norm']:
                keys=['identity_grad_norm','prediction_raw_grad_norm','data_grad_postclip_norm','decay_term_norm','decay_to_data_norm_ratio']
                details.append('| '+family+' | '+' | '.join(fmt(last[k][family]) for k in keys)+' |\n')
    text = '# HD优化诊断自动汇总\n\n短程训练不是完整检索结果；本工具不自动选O，也不把修正量增大判为成功。\n\n'
    text += '| 目录 | 已完成epoch | 日程epoch | 有效gain RMS | Anchor RMS | Corrector RMS | 最大层均修正比例 | AMP跳步 | 梯度记录数 | 正常停止标记 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    text += ''.join('| '+' | '.join(row)+' |\n' for row in rows) + ''.join(details)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:handle.write(text)
    print(output)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
