"""Condition-selected Q/V LoRA, original teacher heads, FP32 text/KD."""
import torch
from torch import nn
import torch.nn.functional as F

class ConditionalQVLora(nn.Module):
    def __init__(self, dim=768, depth=12, modalities=3, platforms=2, rank=8, alpha=8.):
        super().__init__()
        self.dim, self.depth, self.rank = dim, depth, rank
        self.scale = alpha / rank
        for name, banks in [('q_m', modalities), ('v_m', modalities), ('q_p', platforms), ('v_p', platforms)]:
            setattr(self, name + 'A', nn.Parameter(torch.randn(depth, banks, rank, dim) * .02))
            setattr(self, name + 'B', nn.Parameter(torch.zeros(depth, banks, dim, rank)))

    def delta(self, layer, modality, platform):
        def weight(name, condition):
            return getattr(self, name + 'B')[layer, condition] @ getattr(self, name + 'A')[layer, condition] * self.scale
        return weight('q_m', modality) + weight('q_p', platform), weight('v_m', modality) + weight('v_p', platform)

    def qkv_delta(self, layer, modality, platform, x):
        # Apply x A^T B^T. Never allocate [batch,768,768] weight matrices.
        def low_rank(name, condition):
            a = getattr(self, name + 'A')[layer].index_select(0, condition)
            b = getattr(self, name + 'B')[layer].index_select(0, condition)
            return torch.bmm(torch.bmm(x, a.transpose(1, 2)), b.transpose(1, 2)) * self.scale
        q = low_rank('q_m', modality) + low_rank('q_p', platform)
        v = low_rank('v_m', modality) + low_rank('v_p', platform)
        return torch.cat([q, torch.zeros_like(q), v], dim=-1)

class ConditionalVisualTeacher(nn.Module):
    def __init__(self, cfg, num_classes, camera_num=0, view_num=0, load_original=True):
        super().__init__()
        from model import make_model
        c = cfg.clone()
        c.defrost()
        c.M2.ENABLED = False
        c.MODEL.TOKEN_TRAJECTORY = False
        c.MODEL.PRETRAIN_CHOICE = 'imagenet' if load_original else 'no'
        c.MODEL.PRETRAIN_PATH = cfg.M2.CLIP_PATH
        original = make_model(c, num_classes, camera_num, view_num)
        self.base, self.bottleneck, self.classifier = original.base, original.bottleneck, original.classifier
        self.num_classes, self.in_planes, self.neck_feat = num_classes, 768, c.TEST.NECK_FEAT
        for p in self.base.parameters():
            p.requires_grad_(False)
        # Original backbone and heads finish initialization before LoRA draws.
        with torch.random.fork_rng(devices=[]):
            self.lora = ConditionalQVLora(dim=768, depth=12,
                modalities=c.TEACHER_STAGE.MODALITY_BANKS, platforms=c.TEACHER_STAGE.PLATFORM_BANKS,
                rank=c.TEACHER_STAGE.LORA_RANK, alpha=c.TEACHER_STAGE.LORA_ALPHA)

    def forward_pre_bn(self, images, modality, platform):
        return self.base(images, conditional_lora=self.lora,
            conditional_modality=modality, conditional_platform=platform)

    def forward(self, x, label=None, camids=None, mode=0):
        if mode == 0:
            images = torch.cat(list(x), dim=0)
            modality = torch.arange(3, device=images.device).repeat_interleave(len(x[0]))
            cameras = torch.cat(camids).to(images.device)
        else:
            images = x
            modality = torch.full((len(x),), mode - 1, dtype=torch.long, device=x.device)
            cameras = camids.to(x.device)
        platform = ((cameras == 5) | (cameras == 6)).long()
        g = self.forward_pre_bn(images, modality, platform)
        f = self.bottleneck(g)
        if mode == 0:
            return self.classifier(f), g, f
        return f if self.neck_feat == 'after' else g

def normalized_clip_logits(feat, proj, text_bank, tau=.07):
    with torch.cuda.amp.autocast(enabled=False):
        z = F.normalize(feat.float() @ proj.detach().float(), dim=-1)
        return z @ text_bank.detach().float().t() / tau

def kd_kl(logits_teacher, logits_student, temperature=2.):
    with torch.cuda.amp.autocast(enabled=False):
        t = float(temperature)
        lt = F.log_softmax(logits_teacher.detach().float() / t, dim=-1)
        ls = F.log_softmax(logits_student.float() / t, dim=-1)
        return t * t * (lt.exp() * (lt - ls)).sum(-1).mean()
