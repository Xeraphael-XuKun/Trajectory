"""Summarize completed local training evidence; never rank by test-set metrics."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def summarize(root):
    result = {'runs': [], 'diagnostics': []}
    for path in sorted(root.glob('*/result.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        mode = data['args']['mode']; weight = data['args']['weight']
        records = data['records']
        if records:
            def window(rows):
                keys = ('ce','triplet','repair','weighted_repair','ratio_to_mean_main',
                        'ess_fraction','off_soft_error','on_soft_error')
                return {k: mean(r[k] for r in rows if k in r) for k in keys if k in rows[0]}
            norms = [r['grad_norm_before_clip'] for r in records
                     if math.isfinite(r['grad_norm_before_clip'])]
            result['runs'].append({'name': path.parent.name, 'mode':mode, 'weight':weight,
                'steps':len(records), 'amp_skips':sum(r['amp_skipped'] for r in records),
                'clip_fraction_finite':mean(v > 1 for v in norms),
                'mean':window(records), 'first10':window(records[:10]), 'last10':window(records[-10:]),
                'peak_gib':max(r['peak_gib'] for r in records),'elapsed_seconds':data['elapsed_seconds']})
        for diag in data['diagnostics']:
            result['diagnostics'].append({'name':path.parent.name,'weight':weight,**diag})
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('| Run | Steps | CE mean | Triplet mean | Repair mean | Weighted repair | Ratio to mean main | AMP skips |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for run in result['runs']:
        r=run['mean']
        print('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {} |'.format(
            run['name'],run['steps'],r['ce'],r['triplet'],r['repair'],r['weighted_repair'],
            r['ratio_to_mean_main'],run['amp_skips']))
    for d in result['diagnostics']:
        print(json.dumps(d))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); summarize(Path(a.root))
