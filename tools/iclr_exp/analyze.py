"""Chinese result handoff and matched per-query comparisons from saved exports."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def compare(new_dir, old_dir, label, output, filename='per_query_post.jsonl'):
    # Pair by the full evaluation index, never by row number alone.
    for split in ('query','gallery'):
        if read_rows(new_dir/(split+'_index.jsonl'))!=read_rows(old_dir/(split+'_index.jsonl')):
            raise ValueError(label+' 的 '+split+' 索引不一致，不能直接配对')
    new_result, old_result=read_json(new_dir/'results.json'),read_json(old_dir/'results.json')
    if not new_result['formal'] or not old_result['formal']:
        raise ValueError('正式配对比较不得混入小样本评测')
    if new_result['metric']!=old_result['metric'] or new_result['precision']!=old_result['precision']:
        raise ValueError('参考评测协议/精度不匹配')
    new,old=read_rows(new_dir/filename),read_rows(old_dir/filename)
    if len(new)!=len(old):
        raise ValueError('逐query记录数不一致')
    pairs=[]
    for a,b in zip(new,old):
        if a['relative_path']!=b['relative_path'] or a['valid']!=b['valid']:
            raise ValueError('有效query或排序索引不匹配')
        if not a['valid']:continue
        pairs.append(dict(relative_path=a['relative_path'],raw_pid=a['raw_pid'],modality=a['modality'],platform=a['platform'],
            AP_old=b['AP'],AP_new=a['AP'],AP_delta=a['AP']-b['AP'],INP_old=b['INP'],INP_new=a['INP'],
            R1_old=b['R1'],R1_new=a['R1'],first_rank_old=b['first_positive_rank'],first_rank_new=a['first_positive_rank'],
            top1_old=b['top1']['relative_path'],top1_new=a['top1']['relative_path'],
            top1_new_pid=a['top1']['raw_pid'],top1_new_modality=a['top1']['modality'],top1_new_platform=a['top1']['platform']))
    with (output/('paired_'+label+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]) if pairs else ['relative_path'])
        writer.writeheader(); writer.writerows(pairs)
    def summarize(rows):
        delta=np.array([r['AP_delta'] for r in rows])
        def rank_bin(rank):return '1' if rank==1 else ('2_5' if rank<=5 else ('6_10' if rank<=10 else 'above_10'))
        transitions={}
        for row in rows:
            key=rank_bin(row['first_rank_old'])+'_to_'+rank_bin(row['first_rank_new'])
            transitions[key]=transitions.get(key,0)+1
        negative_conditions={};new_negative_conditions={}
        for row in rows:
            key='%d_%d'%(row['top1_new_modality'],row['top1_new_platform'])
            if not row['R1_new']:negative_conditions[key]=negative_conditions.get(key,0)+1
            if row['R1_old'] and not row['R1_new']:new_negative_conditions[key]=new_negative_conditions.get(key,0)+1
        return dict(top1_negative_conditions=negative_conditions,new_wrong_top1_conditions=new_negative_conditions,count=len(rows),improved=int((delta>0).sum()),weakened=int((delta<0).sum()),unchanged=int((delta==0).sum()),
            wrong_to_correct=sum(r['R1_old']==0 and r['R1_new']==1 for r in rows),
            correct_to_wrong=sum(r['R1_old']==1 and r['R1_new']==0 for r in rows),
            remain_correct=sum(r['R1_old']==1 and r['R1_new']==1 for r in rows),
            remain_wrong=sum(r['R1_old']==0 and r['R1_new']==0 for r in rows),
            AP_delta_quantiles=np.quantile(delta,[0,.1,.5,.9,1]).tolist() if len(delta) else None,
            first_positive_rank_transitions=transitions)
    result=dict(ALL=summarize(pairs),query_conditions_full_gallery={})
    for m in range(3):
        for v in range(2):
            result['query_conditions_full_gallery']['%d_%d'%(m,v)]=summarize([r for r in pairs if r['modality']==m and r['platform']==v])
    result['case_selection']='AP_delta descending/ascending; unchanged AP first 20; no visual attribute inference'
    result['improved_examples']=sorted([r for r in pairs if r['AP_delta']>0],key=lambda r:-r['AP_delta'])[:20]
    result['weakened_examples']=sorted([r for r in pairs if r['AP_delta']<0],key=lambda r:r['AP_delta'])[:20]
    result['unchanged_examples']=[r for r in pairs if r['AP_delta']==0][:20]
    return result


def main():
    p=argparse.ArgumentParser(description='生成I1结果交接文档；可选接入匹配的C0/M5特征导出')
    p.add_argument('--evaluation',required=True);p.add_argument('--c0-evaluation');p.add_argument('--m5-evaluation')
    p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.evaluation);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    result=read_json(source/'results.json');run=source.parent
    manifest=read_json(run/'run_manifest.json') if (run/'run_manifest.json').exists() else None
    budget=read_json(run/'stage_budget.json') if (run/'stage_budget.json').exists() else None
    audits={path.stem:read_json(path) for path in sorted((run/'audit').glob('epoch*.json'))}
    gradients={path.stem:read_json(path) for path in sorted(run.glob('gradient_epoch*.json'))}
    comparisons={}
    for label,path in (('C0',a.c0_evaluation),('M5',a.m5_evaluation)):
        comparisons[label]=compare(source,Path(path),label,output) if path else None
        if path and (source/'per_query_G_to_A_post.jsonl').exists() and (Path(path)/'per_query_G_to_A_post.jsonl').exists():
            comparisons[label+'_G_to_A']=compare(source,Path(path),label+'_G_to_A',output,'per_query_G_to_A_post.jsonl')
        for key in ('0_to_2','1_to_2','2_to_0','2_to_1','2_to_2'):
            filename='per_query_mod_'+key+'_post.jsonl'
            if path and (source/filename).exists() and (Path(path)/filename).exists():
                comparisons[label+'_mod_'+key]=compare(source,Path(path),label+'_mod_'+key,output,filename)
    augmented={path.stem:read_json(path) for path in sorted(run.glob('target_augmented_epoch*.json'))}
    stats=read_json(run/'neutral_stats_summary.json') if (run/'neutral_stats_summary.json').exists() else None
    payload=dict(augmented_target_audits=augmented,neutral_crossfit=stats,evaluation=result,manifest=manifest,budget=budget,audits=audits,gradient_probes=gradients,paired_comparisons=comparisons)
    (output/'handoff.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    def metric(value):return 'NA' if value is None else '%.2f'%(100*value)
    neutral=result.get('variant')=='m5_neutral'
    name='I2 M5-Neutral' if neutral else 'I1 C0＋TGVA-Ref'
    lines=['# '+name+' 实验结果交接','',
        '本文件报告实际测量与证据；方法定义请配合实验方案阅读。',
        '','## 执行状态与证据','',
        '- 正式epoch60完整评测：'+('是' if result['formal'] else '否；这是小样本/非最终工件核验，不能作为正式成绩'),
        '- 工件epoch：'+str(result['epoch']),'- 权重：`'+result['checkpoint']+'`',
        '- 权重SHA256：`'+result['checkpoint_sha256']+'`','- 独立严格重载，FP32、SYSU同相机过滤、无rerank；pre/post读取等价检查通过。',
        ('- I2复用冻结旧M5教师、身份文本表与固定训练集偏移；主比较对象为旧M5。' if neutral else '- I1不使用M5教师、身份文本表或中性偏移。'),
        '','## 主结果与完整分组','',
        '| 读出 | mAP | R1 | R5 | R10 | mINP | 有效query |','|---|---:|---:|---:|---:|---:|---:|']
    for neck in ('post','pre'):
        s=result[neck]['ALL'];lines.append('| '+neck+' | '+' | '.join(metric(s[k]) for k in ('mAP','R1','R5','R10','mINP'))+' | '+str(s['valid_queries'])+' |')
    for neck in ('post','pre'):
        for category in ('modality_pairs','platform_pairs','condition_pairs','query_conditions_full_gallery'):
            lines+=['','### '+neck+' / '+category,'','| 方向 | mAP | R1 | R5 | R10 | mINP | 有效query/总query | gallery |','|---|---:|---:|---:|---:|---:|---:|---:|']
            for key,s in result[neck][category].items():
                lines.append('| '+key+' | '+' | '.join(metric(s[k]) for k in ('mAP','R1','R5','R10','mINP'))+' | %d/%d | %d |'%(s['valid_queries'],s['query_count'],s['gallery_count']))
    lines+=['','模态0/1/2为RGB/IR/Thermal，平台0/1为地面/航拍。前三类分组同时限制query/gallery；最后一类保持完整gallery。NA表示无有效query，不能赋零。结构化指标和AP_delta使用0–1；表格乘100显示百分数，百分点差为100×原始差。',
        '','## 机制、预算与运行质量','',
        ('以下记录保留原教师/中性目标/学生的逐图身份信息、交叉拟合、gain、实际修正比、投影漂移、开关特征差和分项梯度。干净面板与固定增强batch分别报告；都不等同于测试检索成功。' if neutral else '以下记录保留实际数值，避免仅凭总体loss或条件准确率判断机制成立。四格任务、gain、实际修正比、锚点余弦、投影漂移、开关特征差与分项梯度均来自固定训练面板/规定batch，不等同于测试检索成功。')]
    for title,value in (('资源与实现manifest',manifest),('运行预算',budget),('固定面板审计',audits),('固定增强面板目标',augmented),('偏移交叉拟合',stats),('分项梯度',gradients)):
        lines+=['','### '+title,'']
        lines+=['未提供，无法确认。'] if value is None or value=={} else ['```json',json.dumps(value,ensure_ascii=False,indent=2),'```']
    lines+=['','## 历史文档参考与重评测差异','', '以下历史值来自第一阶段结果说明，未当成本次重评测：C0 post-BN mAP/R1=13.52/33.61；旧M5=15.01/35.30。']
    for label,path,ma,r1 in (('C0',a.c0_evaluation,13.52,33.61),('M5',a.m5_evaluation,15.01,35.30)):
        if path:
            measured=read_json(Path(path)/'results.json')['post']['ALL']
            lines.append('- '+label+' 重评测 mAP/R1='+metric(measured['mAP'])+'/'+metric(measured['R1'])+'；与文档之差(pp)=%.4f/%.4f。保留差异，不能替换成更有利的数字。'%(100*measured['mAP']-ma,100*measured['R1']-r1))
    lines+=['','## 改善与削弱的query','']
    for label,value in comparisons.items():
        lines+=['','### 对比 '+label,'']
        lines+=['未提供匹配参考导出，未完成逐query配对；不能从聚合指标编造样本对变化。'] if value is None else ['```json',json.dumps(value,ensure_ascii=False,indent=2),'```','完整配对见 `paired_'+label+'.csv`。']
    lines+=['','## 解释边界与待完成工作','',
        ('估计集合的平均空地logits差归零是构造恒等式，不证明身份信息保留、拟合外收益或检索改善。教师弱于旧学生，中性目标可能不可实现。需结合配对/地面-only目标诊断、Thermal和G→A逐query比较。单seed不能证明稳定性。' if neutral else '六锚点目标达标不证明身份信息保留或几何纠正。开关差距增大需排查关闭分支恶化；同模态、Thermal跨模态及G→A应结合完整分组和逐query证据判断。单seed结果不能证明训练稳定性。'),
        '本脚本不自动推断图像遮挡/背景，不自动改变权重、选择最佳epoch或启动额外训练。缺少参考工件和未执行项目如实保留。']
    (output/(('I2' if neutral else 'I1')+'_实验结果说明.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output/(('I2' if neutral else 'I1')+'_实验结果说明.md'))


if __name__=='__main__':main()
