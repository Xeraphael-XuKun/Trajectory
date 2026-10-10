"""从 train 构建原始冻结 CLIP 三光谱缓存，不扫描 query/gallery。"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model.backbones.vit_pytorch import vit_base_clip
from utils.m1_artifacts import PREPROCESS, train_manifest, describe

class TrainImages(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        path, _, _, _ = self.records[i]
        with Image.open(path) as image:
            return self.transform(image.convert('RGB'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--subdir', default='WHU-MARS')
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    records, pid_map = train_manifest(args.data_root, args.subdir)
    meta = describe(records, pid_map, args.clip)
    tfm = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                     T.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])
    loader = DataLoader(TrainImages(records, tfm), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda')
    model = vit_base_clip(img_size=(256, 128), stride_size=[16, 16], drop_path_rate=0)
    model.load_param(args.clip)
    model.requires_grad_(False).to(device).eval()
    features = []
    with torch.no_grad():
        for images in loader:
            features.append(F.normalize(model(images.to(device)).float() @ model.clip_proj.float(), dim=-1).cpu())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({'feature': torch.cat(features), 'pid': torch.tensor([r[1] for r in records]),
                'cams': torch.tensor([r[2] for r in records]),
                'modality': torch.tensor([r[3] for r in records]),
                'platform': torch.tensor([int(r[2] in (5, 6)) for r in records]),
                'metadata': meta}, args.output)
    print('已保存 {} 张 train 图像的冻结教师特征: {}'.format(len(records), args.output))

if __name__ == '__main__':
    main()
