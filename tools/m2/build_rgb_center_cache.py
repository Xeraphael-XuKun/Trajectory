"""Build M2-2 fixed RGB identity centers from a verified train-only cache."""
import argparse
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--rgb', default='RGB')
    args = ap.parse_args()
    obj = torch.load(args.input, map_location='cpu', weights_only=False)
    required = ('features', 'pids', 'cams', 'modalities')
    if not all(k in obj for k in required):
        raise ValueError('input cache must contain features,pids,cams,modalities')
    x = obj['features'].float()
    p = obj['pids'].long()
    c = obj['cams'].long()
    m = obj['modalities']
    names = obj.get('modality_names', ('RGB', 'IR', 'Thermal'))
    if torch.is_tensor(m):
        names = tuple(names)
        rgb_index = names.index(args.rgb) if args.rgb in names else 0
        sel = m.long() == rgb_index
    else:
        sel = torch.tensor([str(v) == args.rgb for v in m], dtype=torch.bool)
    if not sel.any():
        raise ValueError('no RGB rows in input cache; check modality metadata/mapping')
    classes = torch.unique(p[sel]).sort().values
    centers = []
    for y in classes:
        cam_centers = []
        for cam in torch.unique(c[sel & (p == y)]):
            rows = x[sel & (p == y) & (c == cam)]
            cam_centers.append(F.normalize(rows.mean(0, keepdim=True), dim=1)[0])
        centers.append(F.normalize(torch.stack(cam_centers).mean(0, keepdim=True), dim=1)[0])
    torch.save({'centers': torch.stack(centers), 'pids': classes,
                'source': 'train-only RGB, camera-balanced', 'rgb': args.rgb}, args.output)
    print('saved {} centers to {}'.format(len(centers), args.output))


if __name__ == '__main__':
    main()
