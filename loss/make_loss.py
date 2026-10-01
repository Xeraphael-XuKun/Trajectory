import torch.nn.functional as F
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):
    # 保留原 soft-margin batch-hard Triplet；CE 使用同一身份标签空间。
    triplet = TripletLoss()

    def loss_func(score, feat, target):
        identity = F.cross_entropy(score, target)
        metric = triplet(feat, target)[0]
        return (cfg.MODEL.ID_LOSS_WEIGHT * identity +
                cfg.MODEL.TRIPLET_LOSS_WEIGHT * metric), identity, metric

    return loss_func
