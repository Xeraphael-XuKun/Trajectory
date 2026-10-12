"""Result-focused W0/D1/D2 handoff, with live execution states and explicit caveats."""
import argparse
from common import *


def get(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    args=ap.parse_args()
    root=Path(args.root)
    summary=root/'summary'
    summary.mkdir(parents=True,exist_ok=True)
    w,d1,d2=(get(root/folder/file) for folder,file in
              [('W0','artifact_manifest.json'),('D1','manifest.json'),('D2','manifest.json')])
    states={name:doc['status'] if doc else 'NOT_COMPLETED' for name,doc in [('W0',w),('D1',d1),('D2',d2)]}
    for name in states:
        exit_file=root/name/'launcher_exit_code.txt'
        if exit_file.exists():
            code=int(exit_file.read_text().strip())
            if code != 0:
                states[name]='RUNNING_OR_INTERRUPTED' if code == -1 else 'FAILED_EXIT_'+str(code)
    if states['D1'] not in ('COMPLETED','COMPLETED_WITH_CAVEAT','SMOKE_COMPLETED'):
        d1=None
    if states['D2'] not in ('COMPLETED','COMPLETED_WITH_CAVEAT','SMOKE_COMPLETED'):
        d2=None
    write_json(root/'run_status.json',dict(tasks=states,training_tasks='E1/E2/E3 not requested in this execution',
        outputs_root=str(root.resolve()),report_environment=environment(),code_commit=code_commit()))
    text='''## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-10-12
- Verification Status: generated from completion manifests; see task states
- Version Label: r2_diagnostics_v2

# 第二阶段W0、D1、D2执行结果与诊断

仅报告本批资源核验、原模型KD接口及逐query检索分析。E1/E2/E3/D3本次未执行。
所有实际数据与必要口径写入本文；路径只供追溯。事实、机制假设和未验证问题分开报告。

## 1. 完成状态与运行边界

| 任务 | 状态 |
|---|---|
'''
    for name,state in states.items():
        text+=f'| {name} | {state} |\n'
    text+='''
开发分支为codex/m2-r2-diagnostics，从M2-5正式提交db8a5c65ba0b7b5f0c62d00d2adb2f5e3bfe1177派生。
原m2-c/m2-e/C0工件只读，新诊断不训练参数，不改变历史实验结果。正式模式要求原Python3.8、torch2.4.1/cu121、torchvision0.13.1、timm1.0.19及单卡环境。
推理使用FP32、eval/no-grad；正式模式调用原configure_cudnn，保留原torch的TF32默认值。小样本本地模式关闭benchmark和TF32。实际环境、backend开关和完成状态以下列manifest为准，不由本报告生成进程代替。

## 2. W0资源、配置与加载核验

'''
    for name,doc in [('W0',w),('D1',d1),('D2',d2)]:
        if doc:
            text+=f"\n{name}运行环境：`{json.dumps(doc.get('environment'),ensure_ascii=False)}`；正式原环境模式：{doc.get('formal_original_environment',False)}；源码提交：{doc.get('code_commit','未记录')}。\n"
    if w:
        text+='| 角色 | 字节数 | SHA-256 | 元数据摘要 |\n|---|---:|---|---|\n'
        for name,entry in w['artifacts'].items():
            text+=f"| {name} | {entry['bytes']} | {entry.get('sha256','NA')} | {entry.get('metadata',{})} |\n"
        text+=f"\n数据数量：{w.get('data_counts')}。\n核验项：{json.dumps(w.get('checks'),ensure_ascii=False)}。\n"
    text+='''
本地开发时教师原加载器拒绝pos_embed位相等检查。仅此张量的本地原CLIP重算值与保存值不同：max_abs=1.4901161e-8、RMS=1.1022422e-9；其他冻结非fc张量逐元素完全相等。PyTorch版本间插值舍入是候选原因，未在原2.4.1环境验证此因果解释。
诊断兼容分支仅在torch1.13和这个精确错误触发，保留来源/文本/架构/冻结参数检查，要求pos差异max≤2e-8且RMS≤2e-9，然后strict=True载入保存的全部教师state。保存的pos_embed没有改写。该兼容仅在新诊断脚本，原训练/加载器文件未改。
正式模式禁止触发上述兼容，必须通过原教师加载器。旧torch关键字适配及本地runtime_deps不用于服务器正式环境；没有使用strict=False。
实际M5 KD_WEIGHT=0.5、TEXT_TEMPERATURE=0.07、KD_TEMPERATURE=2；M3五个controller张量共186368元素，原LR3.5e-4、WD1e-4。配置路径映射单列，历史配置保留。W0失败项以manifest为准，不声称每张图片字节与云端逐一校验。

## 3. D1：训练身份的真实KD接口

'''
    if d1:
        text+=f"完成训练图像{d1['images']}张；增强子集{d1['augmentation_images']}张×2次，teacher/student均eval，buffers_unchanged={d1['buffers_unchanged']}。耗时{d1['elapsed_seconds']:.1f}秒。\n"
        text+='\n主概率为cosine/0.07后再除Td=2的500类softmax；Td=1单列补充，不混平均。不是测试未知身份分类。\n\n'
        text+='| 指标 | 图像微平均 | 身份宏平均 | 中位数 | p10 | p90 |\n|---|---:|---:|---:|---:|---:|\n'
        for key,entry in d1['overall'].items():
            text+=f"| {key} | {entry['micro_mean']:.8g} | {entry['identity_macro_mean']:.8g} | {entry['median']:.8g} | {entry['p10']:.8g} | {entry['p90']:.8g} |\n"
        text+='\n### 条件分组（Td=2）\n\n'
        text+='| 分组 | 图像/身份 | 教师正确率 | 学生正确率 | aT | aS | aS<aT比例 | 教师尾部相对均匀KL |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
        group=rows(root/'D1/group_summary.csv')
        names=list(dict.fromkeys(r['group'] for r in group))
        for name in names:
            data={r['metric']:r for r in group if r['group']==name}
            entry=next(iter(data.values()))
            text+=f"| {name} | {entry['images']}/{entry['identities']} | "+' | '.join(f"{float(data[k]['micro_mean']):.7g}" for k in
                ('teacher_correct','student_correct','teacher_p_y','student_p_y','active','teacher_tail_uniform_kl'))+' |\n'
        text+='\n### 教师/学生正确性与门控\n\n'
        text+='| 教师正确 | 学生正确 | 图像数 | active比例 |\n|---|---|---:|---:|\n'
        for r in rows(root/'D1/correctness_active_quadrants.csv'):
            text+='| '+' | '.join(r.values())+' |\n'
        aug=rows(root/'D1/augmentation_metrics.csv')
        text+='\n### 有限增强检查\n\n'
        text+='| 指标 | 均值 | p10 | 中位数 | p90 |\n|---|---:|---:|---:|---:|\n'
        for k in ('teacher_delta_p_y','student_delta_p_y','teacher_argmax_flip','teacher_js','active_flip'):
            v=np.array([float(r[k]) for r in aug])
            text+=f'| {k} | {v.mean():.8g} | {np.quantile(v,.1):.8g} | {np.median(v):.8g} | {np.quantile(v,.9):.8g} |\n'
        text+='\n增强采用身份×实际模态×平台每格至多2图、多相机优先，固定清单与两个独立种子，教师学生读取相同增强张量。模型保持eval，不代表训练DropPath/BN行为。完整KD与二元+教师尾部质量×条件尾部KL逐批核对；尾部熵比较ln499，完整熵比较ln500。\n'
    else:
        text+='D1全量结果尚未形成，少量预检不替代全量结果。\n'
    text+='''
## 4. D2：固定权重重评与逐query配对

'''
    metrics=(get(root/'D2/metrics_reloaded.json') or []) if d2 else []
    if metrics:
        text+='| 读出 | 模型 | 协议 | 有效query | mAP% | R1% | R5% | R10% | mINP% |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
        for r in metrics:
            text+=f"| {r['readout']} | {r['model']} | {r['protocol']} | {r['valid_queries']} | "+' | '.join('NA' if r[k] is None else f'{r[k]:.6f}' for k in ('mAP','Rank1','Rank5','Rank10','mINP'))+' |\n'
        write_csv(summary/'main_and_grouped_metrics.csv',metrics)
        hist={'c0':[13.52,33.61,50.05,58.19],'m5':[15.01,35.30,52.08,60.39]}
        text+='\n### 与历史打印值的差异（保留新旧值）\n\n'
        text+='| 模型 | 历史post ALL mAP/R1/R5/R10 | 本轮重评 | 重评−历史/pp |\n|---|---|---|---|\n'
        comparisons=[]
        for role in ('c0','m5') if d2.get('full_test_pass') else ():
            r=next((v for v in metrics if v['readout']=='post' and v['protocol']=='ALL' and v['model']==role),None)
            if r:
                vals=[r[k] for k in ('mAP','Rank1','Rank5','Rank10')]
                text+=f"| {role} | "+'/'.join(map(str,hist[role]))+' | '+'/'.join(f'{v:.6f}' for v in vals)+' | '+'/'.join(f'{v-h:+.6f}' for v,h in zip(vals,hist[role]))+' |\n'
                comparisons.append(dict(model=role,historical=hist[role],reloaded=vals,
                    delta_pp=[v-h for v,h in zip(vals,hist[role])],
                    agrees_at_printed_precision=all(f'{v:.2f}'==f'{h:.2f}' for v,h in zip(vals,hist[role])),
                    formal_full_test=bool(d2.get('formal_original_environment') and d2.get('full_test_pass'))))
        write_json(summary/'historical_comparison.json',comparisons)
        text+='\n仅正式原环境、全测试集且两位小数一致时标记历史口径重现；不一致保留差值，需要排查后才能作主要结论。小样本分数不得用于核对历史整体分数。\n'
    pair=rows(root/'D2/paired_summary.csv') if d2 else []
    if pair:
        text+='\n### 逐query AP与首位变化\n\n'
        text+='| 读出/协议 | 有效query | AP改善/退化/不变 | 错→对 | 对→错 | 均对 | 均错 | ΔAP/pp p10/中位/p90 |\n|---|---:|---|---:|---:|---:|---:|---|\n'
        for r in pair:
            text+=f"| {r['readout']}/{r['protocol']} | {r['valid_queries']} | {r['improved']}/{r['degraded']}/{r['unchanged']} | {r['wrong->correct']} | {r['correct->wrong']} | {r['correct->correct']} | {r['wrong->wrong']} | {r['delta_ap_pp_p10']}/{r['delta_ap_pp_p50']}/{r['delta_ap_pp_p90']} |\n"
    text+='''
AP在CSV中为[0,1]，展示差值×100为pp；RankGain=C0首正例名次−M5首正例名次，正数为改善。标准九模态及平台子组重新限制gallery后完整排序；ALL不由子表平均重建。A→A无有效query记NA。距离以原iter_euclidean_distance_chunks、CPU torch.argsort计算，sysu排除同相机全部gallery；逐query AP/INP/R1与原eval_func逐块交叉核对。原始特征缓存保存pre/post，每次评价再L2归一化。
历史分数只是两位小数打印值；本轮同时记录环境差异、并列距离比例和实际重评结果，不人为修改使其吻合。最近正/负例、首位gallery条件、正负间隔及转换类别均存CSV；未只保存top10，也未常驻完整距离矩阵。
本批没有目视检查真实图片，因此不能由文件名/子组变化认定遮挡、模糊、颜色或背景为原因。具体文件变化可以由error_cases和per_query定位；视觉原因仍未验证。

## 5. 已确认、合理解释与待验证问题

'''
    if d1:
        overall=d1['overall']
        text+=f"事实：训练集教师/学生正确身份概率均值分别为{overall['teacher_p_y']['micro_mean']:.6g}/{overall['student_p_y']['micro_mean']:.6g}，OneSided active比例为{overall['active']['micro_mean']:.6g}。这只描述原模型终点，不能代替E2全程门控曲线。\n\n"
        text+=f"事实：原教师尾部相对均匀KL均值为{overall['teacher_tail_uniform_kl']['micro_mean']:.6g}，二元项/加权尾部项均值为{overall['binary_raw']['micro_mean']:.6g}/{overall['weighted_tail_raw']['micro_mean']:.6g}。尾部具有多少统计结构可以描述，是否有利于未知身份检索必须由E1检验。\n\n"
    text+='''
合理解释：教师检索较弱并不等于其训练身份KD接口无信息；正确身份概率与非目标类条件分布可能发挥不同作用。学生高置信度也不等于检索一定更好。D1/D2观察与E1/E2的机制问题相联系，但未运行新训练，不能作损失因果判断。
尚未验证：E1/E2效果、去衰减后M3控制功能/检索、原环境的逐步等价、训练随机性与统计显著性。既有权重下的query差异不覆盖训练seed变化。

## 6. 失败修正与可追溯输出

开发预检中保留三类问题：完整YAML再merge导致DEVICE_ID字符串被YACS转为int，改为直接读取完整已保存配置；旧环境pos_embed重算位差，采用上述限定诊断兼容；小gallery下原eval_func的max_rank50产生不同长度CMC，交叉检查改用max_rank1，AP/INP仍完整排序，正式R5/R10按首正例名次统计。没有修改原训练、采样、数据、损失或评测源码。

主要索引：W0/artifact_manifest.json、config_*_historical.yml、config_*_local.yml、config_diff_vs_parent.json、optimizer_groups_*_parent.json；D1/train_index.csv、per_image_metrics.csv、group_summary.csv、correctness_active_quadrants.csv、augmentation_subset.csv、augmentation_metrics.csv、augmentation/group_summary.csv、manifest.json；D2/index_query.csv、index_gallery.csv、features/、metrics_reloaded.json、per_query/、paired_summary.csv、error_cases.csv、top1_condition_summary.csv、query_examples.csv、manifest.json。
源代码scripts/r2/全部来自新诊断worktree。资源绝对路径和SHA在W0清单；缓存签名含模型/文本哈希、图像顺序和环境，不能把未完成缓存当成完成。GitHub仅发布源码、资源路径模板、脚本和手册；权重、图像、特征、结果和日志不进入Git，由用户在服务器自行运行。

统计误读检查覆盖11/11：不将受限gallery子表平均作ALL、不从组均值推每张图、不按成功例选择或因结果作条件筛选、不把固定池R1当开放世界概率、不把单seed微小差异当稳定收益、完整报告失败/损失、不从候选优胜推显著性、固定epoch60/post主pre辅、不把关联/后验解释当因果或已验证设计理由。没有增加bootstrap或新综合分。
'''
    (summary/'0.W0_D1_D2执行结果与诊断_1012.md').write_text(text,encoding='utf-8')
    if d1:
        (root/'D1/diagnosis.md').write_text('# D1诊断\n\n状态：'+states['D1']+
            '\n\n主概率Td=2、Td=1单列。完整数值及解释边界见[本轮汇总](../summary/0.W0_D1_D2执行结果与诊断_1012.md)。'+
            '\n\n'+json.dumps(d1['overall'],ensure_ascii=False,indent=2),encoding='utf-8')
    if d2:
        (root/'D2/diagnosis.md').write_text('# D2诊断\n\n状态：'+states['D2']+
            '\n\n完整主/分组指标及配对统计见[本轮汇总](../summary/0.W0_D1_D2执行结果与诊断_1012.md)。'+
            '\n\nquery_examples.csv按ALL/G→A分别选择ΔAP最高五个改善和最低五个退化；未经目视，不推断视觉成因。'+
            '\n\n历史口径核对见summary/historical_comparison.json，原环境全量未完成时不标记重现。',encoding='utf-8')
    print(summary/'0.W0_D1_D2执行结果与诊断_1012.md')


if __name__=='__main__':
    main()
