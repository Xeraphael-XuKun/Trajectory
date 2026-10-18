"""Original frozen CLIP, deterministic train-only three-spectrum cache."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model.backbones.vit_pytorch import vit_base_clip
from tools.m2.artifacts import VERSION, BASE_COMMIT, load_config, train_source, read_clip, seed_stage, code_commit

def build(cfg, device='cuda', batch_size=64):
    seed_stage(cfg)
    rows, source = train_source(cfg)
    net = vit_base_clip(img_size=(256, 128), stride_size=16).to(device).eval()
    net._load_clip_visual(read_clip(cfg.M2.CLIP_PATH))
    for p in net.parameters():
        p.requires_grad_(False)
    tf = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
        transforms.Normalize(source['preprocess']['mean'], source['preprocess']['std'])])
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                with Image.open(row[0]) as image:
                    images.append(tf(image.convert('RGB')))
            g = net(torch.stack(images).to(device)).float()
            out.append(F.normalize(g @ net.clip_proj.float(), dim=1).cpu())
    obj = {'implementation': VERSION, 'clip_features': torch.cat(out),
        'pids': torch.tensor([r[1] for r in rows]), 'camids': torch.tensor([r[2] for r in rows]),
        'modalities': torch.tensor([r[3] for r in rows]), 'platforms': torch.tensor([int(r[2] in (5, 6)) for r in rows]),
        'metadata': {'source': source, 'base_commit': BASE_COMMIT, 'code_commit': code_commit(),
            'config': cfg.dump(), 'optimizer_updates': 0}}
    Path(cfg.M2.FEATURE_CACHE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cfg.M2.FEATURE_CACHE)
    print('train-only cache images=%d identities=%d output=%s' % (len(rows), len(source['pid_map']), cfg.M2.FEATURE_CACHE))
    return obj

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('opts', nargs=argparse.REMAINDER)
    a = p.parse_args()
    build(load_config(a.config_file, a.opts), a.device, a.batch_size)

if __name__ == '__main__':
    main()
