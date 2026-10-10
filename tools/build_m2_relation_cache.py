"""从已核验中心重新构建M2-4余弦关系，不接受无来源裸tensor。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import cfg as defaults
from tools.m2_cache import build_relation_cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--centers')
    parser.add_argument('--output')
    args = parser.parse_args()
    cfg = defaults.clone()
    cfg.merge_from_file(args.config_file)
    cfg.M2.RGB_CENTERS = args.centers or cfg.M2.RGB_CENTERS
    cfg.M2.RELATION_BANK = args.output or cfg.M2.RELATION_BANK
    centers = torch.load(cfg.M2.RGB_CENTERS, map_location='cpu', weights_only=False)
    build_relation_cache(centers, cfg.M2.RELATION_BANK, cfg)
    print('已按真实train来源重新生成关系表')

if __name__ == '__main__':
    main()
