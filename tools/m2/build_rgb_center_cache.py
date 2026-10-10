"""按相机等权聚合 M2-2 固定 RGB 身份中心，保留完整来源。"""
import argparse
import os
import torch
import torch.nn.functional as F

def build_centers(obj):
    if 'metadata' not in obj or obj['metadata']['split'] != 'train':
        raise ValueError('input must be a train-only cache with provenance metadata')
    meta = obj['metadata']
    x, p, c, m = (obj[k] for k in ('features', 'pids', 'cams', 'modalities'))
    rgb_index = meta['modality_names'].index('RGB')
    sel = m == rgb_index
    classes = torch.arange(len(meta['pid_map']))
    if not torch.equal(torch.unique(p[sel]).sort().values, classes):
        raise ValueError('RGB observations must cover all train identity labels')
    if (x.shape != (len(p), 512) or not torch.isfinite(x).all() or
            not all(name.startswith('RGB/') or name.startswith('RGB\\') for name in obj['filenames'])):
        raise ValueError('invalid RGB feature/filename rows')
    x = F.normalize(x.float(), dim=1)
    centers = []
    for y in classes:
        camera_centers = [F.normalize(x[sel & (p == y) & (c == cam)].mean(0), dim=0)
                          for cam in torch.unique(c[sel & (p == y)])]
        centers.append(F.normalize(torch.stack(camera_centers).mean(0), dim=0))
    return {'centers': torch.stack(centers), 'pids': classes,
            'metadata': meta, 'source': 'train-only RGB, camera-balanced'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    centers = build_centers(torch.load(args.input, map_location='cpu', weights_only=False))
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(centers, args.output)
    print('已保存 {} 个中心: {}'.format(len(centers['pids']), args.output))

if __name__ == '__main__':
    main()
