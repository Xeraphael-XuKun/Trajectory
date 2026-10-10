"""核验缓存实际 train 来源；无法补造旧缓存缺失的来源。"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from utils.m1_artifacts import validate_cache

def main():
    parser = argparse.ArgumentParser()
    for key in ('input', 'data-root', 'clip'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--subdir', default='WHU-MARS')
    parser.add_argument('--output')
    args = parser.parse_args()
    cache = torch.load(args.input, map_location='cpu', weights_only=False)
    validate_cache(cache, args.data_root, args.subdir, args.clip)
    if args.output and os.path.abspath(args.output) != os.path.abspath(args.input):
        torch.save(cache, args.output)
    print('缓存实际 train 来源核验通过')

if __name__ == '__main__':
    main()
