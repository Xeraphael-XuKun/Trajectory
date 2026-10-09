"""离线预测器历史依赖检查：真实历史/同模态同视角跨身份置换/常量变化。非检索评估。"""
import argparse
import json
import re
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as T
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg
from model import make_model


def main(args):
    torch.set_num_threads(4)
    c = cfg.clone(); c.merge_from_file(args.config)
    c.MODEL.PRETRAIN_CHOICE = 'no'
    model = make_model(c, 500, 7, 2).to(args.device).eval()
    model.load_param(args.weight)
    adapter = model.base.history_adapter
    transform = T.Compose([T.Resize(c.INPUT.SIZE_TEST), T.ToTensor(),
                           T.Normalize(c.INPUT.PIXEL_MEAN, c.INPUT.PIXEL_STD)])
    # 每个条件选择不同身份，确定性循环置换；不用随机数，不改变模型forward历史。
    conditions = {}
    for modality in c.DATASETS.MODALITIES:
        by_view = {'A': {}, 'G': {}}
        for p in sorted((Path(args.data_root) / args.split / modality).glob('*.jpg')):
            match = re.search(r'(\d+)_c(\d+)', p.name)
            pid, camera = map(int, match.groups())
            view = 'A' if camera - 1 in c.DATASETS.AERIAL_CAMS else 'G'
            by_view[view].setdefault(pid, p)
        for view, identities in by_view.items():
            conditions[modality + '_' + view] = list(identities.items())[:args.identities]
    report = {'checkpoint': args.weight, 'config': args.config, 'device': args.device,
              'split': args.split, 'torch': torch.__version__,
              'constant_reference': 'leave-one-identity-out mean target at each patch position; diagnostic only',
              'conditions': {}}
    for condition, selected in conditions.items():
        if len(selected) < 2:
            report['conditions'][condition] = {'status': '不足两个身份，跳过'}
            continue
        captures, handles = {}, []
        for index, predictor in enumerate(adapter.predictors, 2):
            def hook(module, inputs, output, layer=index):
                # HI2/HF3还会调用一次CLS预测；只记录patch，保持与训练目标一致。
                if inputs[0].shape[1] == adapter.gain.shape[2]:
                    captures[layer] = (inputs[0].detach(), output.detach())
            handles.append(predictor.register_forward_hook(hook))
        images = torch.stack([transform(Image.open(p).convert('RGB')) for _, p in selected]).to(args.device)
        states = {}
        original = adapter.correction
        def capture(h, layer, history, previous_r, collect=False, diagnostics=False):
            result = original(h, layer, history, previous_r, collect, diagnostics)
            current = result[1][-1]
            states[layer + 1] = (current[:, 1:] if adapter.token_scope == 'all' else current).detach()
            return result
        adapter.correction = capture
        with torch.no_grad():
            model(images, mode=1)
        adapter.correction = original
        for handle in handles: handle.remove()
        rows = {}
        with torch.no_grad():
            for layer, (inputs, prediction) in captures.items():
                target = states[layer] - states[layer - 1]
                shuffled = adapter.predictors[layer - 2](inputs.roll(1, 0))
                # 每个位置的其他身份均值，不使用当前身份target，仍不是可部署预测器。
                constant = (target.sum(0, keepdim=True) - target) / (len(selected) - 1)
                mse = lambda x: float((x - target).double().square().mean())
                rows[layer] = {'real_history_mse': mse(prediction), 'shuffled_history_mse': mse(shuffled),
                               'copy_mse': float(target.double().square().mean()), 'loo_mean_change_mse': mse(constant)}
        report['conditions'][condition] = {'identities': [i for i, _ in selected],
                                            'paths': [str(p) for _, p in selected], 'layers': rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--weight', required=True)
    p.add_argument('--data-root', required=True, help='WHU-MARS目录本身，包含query/train')
    p.add_argument('--split', default='query', choices=['query', 'train'])
    p.add_argument('--identities', type=int, default=8)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--output', required=True)
    main(p.parse_args())
