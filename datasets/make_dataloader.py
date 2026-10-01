import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing
from .bases import ImageDataset, ImageDatasetTest
from .sampler import PKMSampler
from .whu_mars import WHU_MARS


def train_collate_fn(batch):
    imgs_ls, pids, camids_ls = zip(*batch)
    n = len(imgs_ls[0])
    imgs = [torch.stack([sample[m] for sample in imgs_ls]) for m in range(n)]
    camids = [torch.tensor([sample[m] for sample in camids_ls], dtype=torch.int64)
              for m in range(n)]
    return imgs, torch.tensor(pids, dtype=torch.int64), camids


def val_collate_fn(batch):
    imgs, pids, camids, modids, paths = zip(*batch)
    return torch.stack(imgs), pids, camids, torch.tensor(camids), modids, paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB), T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN), T.ToTensor(),
        T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
        T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])
    if cfg.DATASETS.NAMES != 'WHU-MARS' or cfg.DATALOADER.SAMPLER != 'PKM':
        raise ValueError('this project uses WHU-MARS and PKM')
    mods = list(cfg.DATASETS.MODALITIES)
    dataset = WHU_MARS(root=cfg.DATASETS.ROOT_DIR, modalities=mods,
                       protocol=cfg.DATASETS.PROTOCOL, subdir=cfg.DATASETS.SUBDIR)
    train_loader = DataLoader(
        ImageDataset(dataset.train, mods, train_transforms),
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=PKMSampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                           cfg.DATALOADER.NUM_INSTANCE, mods),
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn)
    val_loaders, num_querys = [], []
    for mod in mods:
        data = {mod: dataset.query[mod] + dataset.gallery[mod]}
        val_loaders.append(DataLoader(
            ImageDatasetTest(data, mod, val_transforms),
            batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn))
        num_querys.append(len(dataset.query[mod]))
    return train_loader, val_loaders, num_querys, dataset.num_train_pids
