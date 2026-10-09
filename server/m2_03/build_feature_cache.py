"""Build the M2-3 stage-A cache from train images only.

The cache uses the deterministic 256x128 CLIP-normalised path and records PID,
modality, and ground/aerial condition metadata.  Query/gallery are never read.
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets.whu_mars import WHU_MARS
from datasets.bases import read_image
from model.backbones.vit_pytorch import vit_base_clip

class TrainImages(Dataset):
    def __init__(self, records, transform): self.records, self.transform = records, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        path, pid, cam, mod = self.records[i]
        return self.transform(read_image(path)), pid, cam, mod

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--clip', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    ds = WHU_MARS(root=a.data_root, modalities=['RGB', 'IR', 'Thermal'])
    records = [row for mod in ['RGB', 'IR', 'Thermal'] for row in ds.train[mod]]
    tfm = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(),
                     T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])])
    loader = DataLoader(TrainImages(records, tfm), batch_size=a.batch_size,
                        shuffle=False, num_workers=a.workers, pin_memory=True)
    model = vit_base_clip(img_size=(256, 128), stride_size=16, drop_path_rate=0)
    model.load_param(a.clip); model.eval().cuda()
    feats, labels, modalities, platforms = [], [], [], []
    with torch.no_grad():
        for images, pid, cam, mod in loader:
            f = F.normalize(model(images.cuda()).float() @ model.clip_proj.float(), dim=-1)
            feats.append(f.cpu()); labels.append(pid); modalities.append(mod)
            platforms.append((cam >= 5).long())
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    torch.save({'features': torch.cat(feats), 'labels': torch.cat(labels),
                'modality': torch.cat(modalities),
                'platform': torch.cat(platforms),
                'source': 'WHU-MARS train only, deterministic CLIP path'}, a.output)
    print('saved {} features to {}'.format(sum(x.shape[0] for x in feats), a.output))

if __name__ == '__main__': main()
