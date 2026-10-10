"""生成 M2-2 的 train-only、原始 CLIP RGB 教师缓存。"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from PIL import Image
from torchvision import transforms
from model.backbones.vit_pytorch import vit_base_clip
from utils.m2_artifacts import PREPROCESS, train_manifest, metadata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    device = torch.device(a.device)
    rows, pid_map = train_manifest(a.data_root)
    meta = metadata(rows, pid_map, ['RGB', 'IR', 'Thermal'], a.clip)
    rgb_rows = [r for r in rows if r[3] == 0]
    if {r[1] for r in rgb_rows} != set(pid_map.values()):
        raise ValueError('some train identities have no RGB observations')
    model = vit_base_clip(img_size=(256, 128), stride_size=[16, 16],
                          drop_path_rate=0.0).to(device).eval()
    model.load_param(a.clip)
    model.requires_grad_(False)
    tfm = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                              transforms.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])
    feats = []
    with torch.no_grad():
        for start in range(0, len(rgb_rows), a.batch_size):
            images = []
            for path, _, _, _ in rgb_rows[start:start + a.batch_size]:
                with Image.open(path) as im:
                    images.append(tfm(im.convert('RGB')))
            batch = torch.stack(images).to(device)
            feats.append(torch.nn.functional.normalize(
                model(batch).float() @ model.clip_proj.float(), dim=1).cpu())
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    torch.save({'features': torch.cat(feats),
                'pids': torch.tensor([r[1] for r in rgb_rows]),
                'cams': torch.tensor([r[2] for r in rgb_rows]),
                'modalities': torch.tensor([r[3] for r in rgb_rows]),
                'filenames': [str(Path('RGB') / Path(r[0]).name) for r in rgb_rows],
                'platform': torch.tensor([int(r[2] in (5, 6)) for r in rgb_rows]),
                'metadata': meta}, a.output)
    print('已保存 train-only RGB 缓存:', a.output, 'rows=', len(rgb_rows))

if __name__ == '__main__':
    main()
