"""Validate and package a train-only CLIP feature cache for M2-1."""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    x = torch.load(a.input, map_location='cpu', weights_only=False)
    for k in ('feature', 'pid', 'modality', 'platform'):
        if k not in x:
            raise ValueError('cache missing ' + k)
    if len(x['feature']) != len(x['pid']) or len(x['pid']) != len(x['modality']):
        raise ValueError('cache lengths differ')
    if x['pid'].min() < 0 or x['modality'].min() < 0 or x['modality'].max() > 2:
        raise ValueError('invalid train mapping')
    out = dict(x)
    for k in ('feature', 'pid', 'modality', 'platform'):
        if hasattr(out[k], 'cpu'):
            out[k] = out[k].cpu()
    torch.save(out, a.output)


if __name__ == '__main__':
    main()
