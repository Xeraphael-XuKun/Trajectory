"""D3: fixed real inputs, native source, same checkpoint controller bypass.

This is an inference function diagnostic, not a training ablation or retrieval
evaluation. Only a controller return hook changes the bypass forward.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_train(root):
    # Identical global index / PID / camera convention to D1 common.index_split.
    rows = []
    for mod, name in enumerate(('RGB', 'IR', 'Thermal')):
        for p in sorted((root / 'train' / name).glob('*.jpg')):
            pid, camera = map(int, re.search(r'(\d+)_c(\d+)', p.name).groups())
            rows.append(dict(index=len(rows), relative_path=p.relative_to(root).as_posix(),
                             raw_pid=pid, camid=camera-1, modality=mod,
                             modality_name=name, platform=int(camera-1 in (5, 6))))
    mapping = {p: y for y, p in enumerate(sorted({r['raw_pid'] for r in rows}))}
    for row in rows:
        row['train_pid'] = mapping[row['raw_pid']]
    assert len(rows) == 92133 and len(mapping) == 500
    return rows, mapping


def select_manifest(args):
    rows, _ = index_train(Path(args.dataset))
    cells = defaultdict(list)
    for row in rows:
        cells[row['train_pid'], row['modality'], row['platform']].append(row)
    complete = [p for p in range(500) if all((p, m, v) in cells for m in range(3) for v in range(2))]
    rng = random.Random(args.seed)
    rng.shuffle(complete)
    count = args.images // 6
    assert 1 <= count <= len(complete) and args.images <= 512
    selected = []
    for p in complete[:count]:
        for m in range(3):
            for v in range(2):
                selected.append(dict(cells[p, m, v][rng.randrange(len(cells[p, m, v]))], probe_index=len(selected)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'fixed_train_images.csv', selected)
    write_json(output / 'selection.json', dict(seed=args.seed, images=len(selected),
        identities=count, eligible_six_condition_identities=len(complete), train_images=len(rows),
        rule='Seeded shuffle of identities with all six modality/platform cells; one seeded random image per cell',
        index_convention='D1 scripts/r2/common.py:index_split(train)',
        camera_counts=dict(Counter(str(r['camid']) for r in selected)),
        condition_counts=dict(Counter('%s_%s' % (r['modality_name'], r['platform']) for r in selected)),
        limitation='Balanced diagnostic subset restricted to identities with all six conditions; not a population performance estimate'))
    print('D3_MANIFEST', len(selected), 'images', count, 'identities', flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Moments:
    """FP64 parallel Welford moments across ALL images at each token/channel."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None
        self.minimum, self.maximum = math.inf, -math.inf
        self.entropy_sum = self.entropy_n = 0
        self.delta2 = self.correction2 = 0.0
        self.bound_n = self.element_n = 0
        self.attention_peak_sum = 0.0

    def add(self, multiplier, attention, correction, output):
        x = multiplier.detach().cpu().double()
        n, mean = x.shape[0], x.mean(0)
        m2 = ((x - mean) ** 2).sum(0)
        if self.n:
            delta = mean - self.mean
            total = self.n + n
            self.m2 += m2 + delta.square() * (self.n * n / total)
            self.mean += delta * (n / total)
        else:
            self.mean, self.m2 = mean, m2
        self.n += n
        self.minimum = min(self.minimum, x.min().item())
        self.maximum = max(self.maximum, x.max().item())
        self.bound_n += ((x-1).abs() >= .249).sum().item()
        self.element_n += x.numel()
        a = attention.detach().double()
        entropy = -(a * a.clamp_min(1e-300).log()).sum(-1)
        self.entropy_sum += entropy.sum().item()
        self.entropy_n += entropy.numel()
        self.attention_peak_sum += a.max(-1).values.sum().item()
        c, d = correction.double(), output.double() - correction.double()
        self.delta2 += d.square().sum().item()
        self.correction2 += c.square().sum().item()

    def result(self, layer):
        input_var = (self.m2 / self.n).mean().item()
        static_var = self.mean.var(unbiased=False).item()
        return dict(layer=layer, block=layer+1, images=self.n,
            multiplier_mean=self.mean.mean().item(), multiplier_std=math.sqrt(input_var + static_var),
            multiplier_min=self.minimum, multiplier_max=self.maximum,
            multiplier_max_abs_from_one=max(abs(self.minimum-1), abs(self.maximum-1)),
            near_bound_fraction=self.bound_n/self.element_n,
            near_bound_threshold=.249,
            input_var=input_var, input_std=math.sqrt(input_var),
            across_token_channel_mean_var=static_var,
            attention_entropy=self.entropy_sum / self.entropy_n,
            attention_max_probability_mean=self.attention_peak_sum/self.entropy_n,
            uniform_attention_entropy=math.log(4),
            correction_delta_l2=math.sqrt(self.delta2), correction_l2=math.sqrt(self.correction2),
            rel_change=math.sqrt(self.delta2) / (math.sqrt(self.correction2) + 1e-12))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def feature_rows(rows, features):
    result = []
    for readout in ('pre', 'post'):
        on, bypass = features[readout + '_on'].double(), features[readout + '_bypass'].double()
        diff = on - bypass
        l2 = diff.norm(dim=1)
        # Half squared chord distance = 1-cosine, stable for identical vectors.
        unit_delta = F.normalize(on, dim=1) - F.normalize(bypass, dim=1)
        cos_delta = .5 * unit_delta.square().sum(1)
        for i, row in enumerate(rows):
            result.append(dict(row, readout=readout, abs_l2=l2[i].item(),
                abs_max=diff[i].abs().max().item(), rel_l2=(l2[i]/(on[i].norm()+1e-12)).item(),
                cosine_difference=cos_delta[i].item(), unit_l2=unit_delta[i].norm().item()))
    return result


def probe(args):
    start = time.monotonic()
    source = Path(args.source_repo).resolve()
    sys.path.insert(0, str(source))
    from config import cfg as defaults
    from yacs.config import CfgNode
    from model import make_model
    import timm
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    if args.run_location == 'server':
        assert sys.version_info[:2] == (3, 8), 'Use the original llmpar Python 3.8 environment'
        assert torch.__version__.startswith('2.4.1') and torch.version.cuda == '12.1'
        assert torchvision.__version__.startswith('0.13.1') and timm.__version__ == '1.0.19'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    obj = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta, state = obj['metadata'], obj['state_dict']
    assert meta['epoch'] == 60 and meta['variant'] == 'deep_text_c0_control'
    assert meta['implementation'] == 'm2-3-audit-v2'
    source_changes = subprocess.check_output(['git', '-C', str(source), 'diff',
        meta['code_commit'], '--', 'model', 'config', 'utils'], text=True)
    assert not source_changes.strip(), 'Native model/config/utils differ from checkpoint training source'
    c = CfgNode.load_cfg(meta['config'])
    def missing(target, ref):
        for k, v in ref.items():
            if k not in target:
                target[k] = v.clone() if isinstance(v, CfgNode) else v
            elif isinstance(v, CfgNode):
                missing(target[k], v)
    missing(c, defaults)
    assert c.M2.ENABLED and c.MODEL.TOKEN_TRAJECTORY
    assert not (c.MODEL.MOD_DELTA or c.MODEL.VPR or c.MODEL.SIE_CAMERA or c.MODEL.SIE_VIEW)
    assert c.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX == 0 and c.M2.CONTROL_RHO == .25
    expected_r2 = meta.get('r2')
    if args.role == 'e3':
        assert expected_r2 == dict(experiment='m3_no_weight_decay',
            parent_code_commit='a432df3cc59c92aab91afdacd43ea1bf81d522fd', controller_weight_decay=0.0)
    else:
        assert expected_r2 is None
    rows_all, pid_map = index_train(Path(args.dataset))
    teacher = meta['teacher']
    assert teacher['pid_map'] == pid_map
    assert teacher['filenames'] == [r['relative_path'][6:] for r in rows_all]
    rows = list(csv.DictReader(Path(args.manifest).open(encoding='utf-8-sig')))
    assert 0 < len(rows) <= 512
    for row in rows:
        for k in ('index', 'raw_pid', 'camid', 'modality', 'platform', 'train_pid', 'probe_index'):
            row[k] = int(row[k])
        assert all(rows_all[row['index']][k] == v for k, v in row.items() if k != 'probe_index')
    counts = Counter((r['modality'], r['platform']) for r in rows)
    assert len(counts) == 6 and len(set(counts.values())) == 1
    c.MODEL.PRETRAIN_CHOICE = 'no'
    c.M2.DEPLOY_ONLY = True
    c.freeze()
    torch.set_num_threads(2)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = make_model(c, num_class=len(pid_map), camera_num=7, view_num=3)
    # Native metadata validator plus strict loading INCLUDING classifier.
    model.load_param(args.checkpoint)
    model.load_state_dict(state, strict=True)
    full_match = None
    if args.full_checkpoint:
        full = torch.load(args.full_checkpoint, map_location='cpu', weights_only=False)
        full_state = full['state_dict']
        assert full['metadata']['epoch'] == meta['epoch']
        assert full['metadata']['config'] == meta['config']
        assert all(k in full_state and torch.equal(v, full_state[k]) for k, v in state.items())
        full_match = dict(deployment_tensors=len(state), all_exactly_equal=True)
        del full, full_state
    del obj, state
    if args.check_only:
        write_json(output/'loading_check.json', dict(status='STRICT_LOADING_CHECK_COMPLETED', role=args.role,
            checkpoint_epoch=meta['epoch'], strict_all_state_loading=True,
            native_model_config_utils_match_training_commit=True, full_deployment_equality=full_match,
            images_forwarded=0, training_updates=0))
        print('D3_LOADING_CHECK_COMPLETE', args.role, 'no image forward', flush=True)
        return
    model.to(args.device).eval()
    controller = model.base.deep_c0_controller
    if hasattr(controller, 'r2_collect_stats'):
        controller.r2_collect_stats = False
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    buffers_before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
    parameters = [{ 'name': n, 'elements': p.numel(), 'l2_fp64': p.detach().double().norm().item(),
                    'abs_max': p.detach().abs().max().item()} for n, p in controller.named_parameters()]
    assert len(parameters) == 5 and sum(p['elements'] for p in parameters) == 186368
    write_json(output/'controller_parameters.json', parameters)
    transform = T.Compose([T.Resize(c.INPUT.SIZE_TEST), T.ToTensor(),
                            T.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD)])
    accum = {l: Moments() for l in range(1, 12)}
    calls = Counter()
    mode = ['on']
    def hook(module, inputs, result):
        layer, velocity, correction, kv = inputs
        calls[mode[0], int(layer)] += 1
        if mode[0] == 'bypass':
            return correction
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            key, value = kv
            u = F.layer_norm(velocity.float(), (velocity.shape[-1],))
            attention = torch.softmax(module.q(u) @ key[int(layer)-1].t() / module.q.out_features**.5, dim=-1)
            multiplier = 1 + module.rho * torch.tanh(module.o(attention @ value[int(layer)-1]))
            assert torch.equal(result, correction * multiplier.to(correction.dtype))
            accum[int(layer)].add(multiplier, attention, correction, result)
        return None
    handle = controller.register_forward_hook(hook)
    features = {k: [] for k in ('pre_on', 'post_on', 'pre_bypass', 'post_bypass')}
    with torch.no_grad():
        for first in range(0, len(rows), args.batch_size):
            images = []
            for row in rows[first:first+args.batch_size]:
                with Image.open(Path(args.dataset)/row['relative_path']) as im:
                    images.append(transform(im.convert('RGB')))
            x = torch.stack(images).to(args.device)
            for branch in ('on', 'bypass'):
                mode[0] = branch
                pre = model.base(x)
                post = model.bottleneck(pre)
                assert torch.isfinite(pre).all() and torch.isfinite(post).all()
                features['pre_' + branch].append(pre.cpu())
                features['post_' + branch].append(post.cpu())
            if first == 0 or (first + args.batch_size) % 60 == 0 or first + args.batch_size >= len(rows):
                print('D3_PROGRESS', args.role, min(first+args.batch_size, len(rows)), '/', len(rows), flush=True)
    handle.remove()
    assert all(not m.training for m in model.modules())
    state_unchanged = all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
    buffers_unchanged = all(torch.equal(v.cpu(), buffers_before[k]) for k, v in model.named_buffers())
    assert state_unchanged and buffers_unchanged
    batches = math.ceil(len(rows)/args.batch_size)
    assert all(calls[b, l] == batches and accum[l].n == len(rows) for b in ('on', 'bypass') for l in range(1, 12))
    features = {k: torch.cat(v) for k, v in features.items()}
    torch.save(features, output/'features_on_bypass.pt')
    layer_rows = [accum[l].result(l) for l in range(1, 12)]
    write_csv(output/'layer_statistics.csv', layer_rows)
    per_image = feature_rows(rows, features)
    write_csv(output/'per_image_feature_difference.csv', per_image)
    summaries = []
    for readout in ('pre', 'post'):
        for group in ['all'] + ['%s_%s' % (m, v) for m in ('RGB', 'IR', 'Thermal') for v in (0, 1)]:
            subset = [r for r in per_image if r['readout'] == readout and
                      (group == 'all' or '%s_%s' % (r['modality_name'], r['platform']) == group)]
            summaries.append(dict(readout=readout, group=group, images=len(subset),
                **{k: stats([r[k] for r in subset]) for k in ('abs_l2', 'abs_max', 'rel_l2', 'cosine_difference', 'unit_l2')}))
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    summary = dict(status='COMPLETED_FIXED_INPUT_GPU_PROBE' if args.device.startswith('cuda') else 'COMPLETED_FIXED_INPUT_CPU_PROBE',
        run_location=args.run_location,
        role=args.role, checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_epoch=meta['epoch'], training_code_commit=meta.get('code_commit'), source_repo=str(source), source_commit=commit,
        probe_code=str(Path(__file__).resolve()), probe_code_sha256=digest(__file__),
        manifest=str(Path(args.manifest).resolve()), manifest_sha256=digest(args.manifest), images=len(rows),
        transform=dict(size=list(c.INPUT.SIZE_TEST), interpolation='bilinear', mean=list(c.INPUT.PIXEL_MEAN), std=list(c.INPUT.PIXEL_STD)),
        inference='float32 eval no_grad; deterministic cuDNN; TF32 disabled; two forwards of each identical input batch',
        batch_size=args.batch_size, epsilon=1e-12, state_unchanged=state_unchanged, buffers_unchanged=buffers_unchanged,
        strict_all_state_loading=True, full_deployment_equality=full_match,
        native_model_config_utils_match_training_commit=True,
        bypass='forward hook returns incoming correction only; original C0/gain and BN retained',
        cosine_difference_definition='0.5 * ||normalize(f_on)-normalize(f_bypass)||^2 in FP64',
        input_var_definition='mean_t,d Var_i(M_i,l,t,d), population variance across ALL manifest images via FP64 Welford',
        rel_change_definition='global Frobenius norm across all images/tokens/channels, not mean batch ratios',
        feature_relative_denominator='norm(f_on) + 1e-12, per image',
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__,
            timm=timm.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None),
        feature_summaries=summaries, layers=layer_rows, elapsed_seconds=time.monotonic()-start,
        limitations=['Not a full test-set retrieval evaluation', 'Not an independent training ablation',
                    'Runtime recorded separately; local inference does not reproduce the historical server runtime',
                    'Subset restricted to complete six-condition identities'])
    write_json(output/'summary.json', summary)
    print('D3_COMPLETE', args.role, len(rows), round(summary['elapsed_seconds'], 2), flush=True)


def summarize(args):
    root = Path(args.output)
    reports = {role: json.loads((root/role/'summary.json').read_text(encoding='utf-8'))
               for role in ('original', 'e3')}
    old, new = reports['original'], reports['e3']
    for k in ('manifest_sha256', 'images', 'transform', 'inference', 'batch_size', 'run_location'):
        assert old[k] == new[k], 'D3 comparison mismatch: ' + k
    assert all(r['state_unchanged'] and r['buffers_unchanged'] and
               r['strict_all_state_loading'] for r in reports.values())
    scope = '服务器固定输入D3探针完成' if new['run_location'] == 'server' else '本地脚本核验，不能作为最终D3结果'
    lines = ['# D3：同权重控制开启/旁路诊断', '', '状态：' + scope + '。', '',
             '两组epoch60使用同一%d张条件平衡训练图像清单。模型eval/no-grad，FP32推理；旁路只返回控制器输入的correction。原C0、gain、BN和全部state未改变。' % new['images'], '',
             '| 模型 | readout | 平均绝对L2差 | 平均相对L2差 | 平均单位向量余弦差 | 最大相对L2差 |',
             '|---|---|---:|---:|---:|---:|']
    for role, r in reports.items():
        for s in r['feature_summaries']:
            if s['group'] == 'all':
                lines.append('| %s | %s | %.6g | %.6g | %.6g | %.6g |' %
                    (role, s['readout'], s['abs_l2']['mean'], s['rel_l2']['mean'],
                     s['cosine_difference']['mean'], s['rel_l2']['maximum']))
    lines.extend(['', '| 模型 | 层/视觉block | InputVar | RelChange | attention熵 | 近边界比例 |',
                  '|---|---|---:|---:|---:|---:|'])
    for role, r in reports.items():
        for s in r['layers']:
            lines.append('| %s | %d/%d | %.6g | %.6g | %.6g | %.4f%% |' %
                (role, s['layer'], s['block'], s['input_var'], s['rel_change'],
                 s['attention_entropy'], 100*s['near_bound_fraction']))
    lines.extend(['', 'InputVar对全部图像、相同token/通道以FP64 Welford统计；RelChange为全部图像的Frobenius范数比。近边界定义为|M−1|≥0.249，理论范围0.75至1.25；四槽均匀attention熵为ln4≈1.386294。', '',
        '解释边界：输入相关乘子、非零参数、较大特征差只能说明终点功能状态，不能证明语义有效或检索获益。若近边界比例高且熵很低，需同时考虑tanh饱和/槽选择过于集中；不能直接断言它是性能变化的原因。旧模型终点旁路近似不变，也不能否定它早期参与训练的影响。', '',
        '没有进行全测试集旁路评测，也没有新训练；E3正式检索成绩仍使用控制开启模型。清单只覆盖具备完整六条件的%d个身份，不用于估计全训练集或测试集总体效果。' % (new['images']//6), '',
        '中期checkpoint：本次未提供/未检查；本批仅诊断两组epoch60部署checkpoint。', '',
        '证据：fixed_train_images.csv、selection.json；各角色summary.json、layer_statistics.csv、controller_parameters.json、per_image_feature_difference.csv、features_on_bypass.pt；run.log。', ''])
    for role, r in reports.items():
        lines.extend(['- %s：权重 `%s`，SHA256 `%s`；训练源码提交 `%s`，本次原生源码提交 `%s`。' %
                      (role, r['checkpoint'], r['checkpoint_sha256'], r['training_code_commit'], r['source_commit'])])
    write_json(root/'comparison.json', dict(status=scope, original=old, e3=new))
    (root/'D3_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('D3_SUMMARY_COMPLETE', scope, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select')
    select.add_argument('--dataset', required=True, help='WHU-MARS root containing train/')
    select.add_argument('--output', required=True)
    select.add_argument('--images', type=int, default=510)
    select.add_argument('--seed', type=int, default=1234)
    run = sub.add_parser('probe')
    run.add_argument('--role', choices=('original', 'e3'), required=True)
    run.add_argument('--source-repo', required=True, help='Native model source checkout; one role per process')
    run.add_argument('--checkpoint', required=True)
    run.add_argument('--full-checkpoint', help='Optional exact full/deployment tensor equality check')
    run.add_argument('--dataset', required=True)
    run.add_argument('--manifest', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--device', default='cuda')
    run.add_argument('--run-location', choices=('local', 'server'), default='local')
    run.add_argument('--check-only', action='store_true', help='Strict checkpoint/config/manifest loading only; no image forward')
    run.add_argument('--batch-size', type=int, default=6)
    summary = sub.add_parser('summarize')
    summary.add_argument('--output', required=True, help='Directory containing original/ and e3/ probe outputs')
    args = ap.parse_args()
    if args.command == 'select':
        select_manifest(args)
    elif args.command == 'probe':
        probe(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
