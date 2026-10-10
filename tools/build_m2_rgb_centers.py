"""M2-4：原始冻结CLIP train/RGB特征→相机均衡中心→余弦关系。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config import cfg as defaults
from model.backbones.vit_pytorch import vit_base_clip
from utils.m4_artifacts import train_manifest, describe, PREPROCESS, VERSION, BASE_COMMIT, file_hash, validate_features, camera_balanced_centers
from tools.m2_cache import build_relation_cache

class RGBImages(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
            transforms.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        with Image.open(self.rows[i][0]) as image:
            return self.transform(image.convert('RGB'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    for key in ('data-root', 'clip', 'output', 'relation-output', 'feature-output'):
        parser.add_argument('--' + key)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = defaults.clone()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.ROOT_DIR = args.data_root or cfg.DATASETS.ROOT_DIR
    cfg.M2.CLIP_PATH = args.clip or cfg.M2.CLIP_PATH
    cfg.MODEL.PRETRAIN_PATH = cfg.M2.CLIP_PATH
    cfg.M2.RGB_CENTERS = args.output or cfg.M2.RGB_CENTERS
    cfg.M2.RELATION_BANK = args.relation_output or cfg.M2.RELATION_BANK
    cfg.M2.FEATURE_CACHE = args.feature_output or cfg.M2.FEATURE_CACHE
    rows, mapping = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)
    rgb = [row for row in rows if row[3] == 0]
    missing = sorted(set(range(len(mapping))) - {row[1] for row in rgb})
    if missing:
        raise ValueError('train identities without RGB; no centers fabricated: {}'.format(missing))
    device = torch.device(args.device)
    torch.manual_seed(cfg.SOLVER.SEED)
    model = vit_base_clip(img_size=(256,128), stride_size=[16,16], drop_path_rate=0.)
    model.load_param(cfg.M2.CLIP_PATH)
    model.requires_grad_(False).to(device).eval()
    loader = DataLoader(RGBImages(rgb), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    features = []
    with torch.no_grad():
        for images in loader:
            features.append(F.normalize(model(images.to(device)).float() @ model.clip_proj.float(), dim=1).cpu())
    meta = describe(rows, mapping, cfg.M2.CLIP_PATH)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
    common = {'variant': 'clip_relation_distill', 'implementation': VERSION, 'base_commit': BASE_COMMIT,
              'code_commit': commit, 'config': cfg.dump(), 'metadata': meta,
              'preparation_optimizer_updates': 0, 'teacher_feature_images': len(rgb)}
    cache = dict(common, features=torch.cat(features), pid=torch.tensor([row[1] for row in rgb]),
                 cams=torch.tensor([row[2] for row in rgb]), modality=torch.tensor([row[3] for row in rgb]),
                 platform=torch.tensor([int(row[2] in (5,6)) for row in rgb]))
    validate_features(cache, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.M2.FEATURE_CACHE)), exist_ok=True)
    torch.save(cache, cfg.M2.FEATURE_CACHE)
    centers, camera_rows = camera_balanced_centers(cache['features'], cache['pid'], cache['cams'], len(mapping))
    center_artifact = dict(common, centers=centers, pid_order=list(range(len(mapping))), camera_rows=camera_rows,
        center_recipe='image_l2_camera_mean_l2_equal_camera_mean_l2', feature_sha256=file_hash(cfg.M2.FEATURE_CACHE))
    os.makedirs(os.path.dirname(os.path.abspath(cfg.M2.RGB_CENTERS)), exist_ok=True)
    torch.save(center_artifact, cfg.M2.RGB_CENTERS)
    build_relation_cache(center_artifact, cfg.M2.RELATION_BANK, cfg)
    print('已保存{}张train/RGB特征、{}个相机均衡身份中心和原始余弦关系；没有prompt或教师训练。'.format(len(rgb),len(mapping)), flush=True)

if __name__ == '__main__':
    main()
