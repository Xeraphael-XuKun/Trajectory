import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_clip


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.num_classes = num_classes
        self.modalities = list(cfg.DATASETS.MODALITIES)
        self.use_token_trajectory = cfg.MODEL.TOKEN_TRAJECTORY
        self.base = vit_base_clip(
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            token_trajectory=self.use_token_trajectory,
            token_trajectory_accel_mix=cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX)
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
        elif cfg.MODEL.PRETRAIN_CHOICE != 'no':
            raise ValueError('PRETRAIN_CHOICE must be imagenet or no')
        if self.use_token_trajectory:
            print('Dense Cross-layer Token Trajectory: accel_mix {}, {:,} params'.format(
                cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX,
                self.base.token_trajectory.trajectory_parameters))
        # 与原单一身份分类器保持相同构造、初始化顺序。
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.text_align = cfg.MODEL.TEXT_ALIGN
        self.prompts = None
        if self.text_align:
            if not self.use_token_trajectory:
                raise ValueError('reserved VTC requires TOKEN_TRAJECTORY=True')
            from .backbones.clip_text import CLIPTextEncoder, ViewPrompts
            clip_sd = torch.load(cfg.MODEL.TEXT_CLIP_PATH, map_location='cpu', weights_only=False)
            if isinstance(clip_sd, nn.Module):
                clip_sd = clip_sd.state_dict()
            text = CLIPTextEncoder()
            text.load_clip(clip_sd)
            self.prompts = ViewPrompts(text, template=cfg.MODEL.TEXT_TEMPLATE,
                                       n_ctx=cfg.MODEL.TEXT_N_CTX)
            self.text_modality_index = self.modalities.index(cfg.MODEL.TEXT_MODALITY)
            self.register_buffer('aerial_cams', torch.tensor(cfg.DATASETS.AERIAL_CAMS).long(), persistent=False)
            scale = clip_sd.get('logit_scale')
            self.register_buffer('logit_scale', scale.exp().float() if scale is not None
                                 else torch.tensor(100.0), persistent=False)

    def forward(self, x, label=None, camids=None, mode=0):
        if mode == 0:
            imgs = list(x)
            per_modality = imgs[0].shape[0]
            x = torch.cat(imgs, dim=0)
            global_feat = self.base(x)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            if not self.text_align:
                return cls_score, global_feat, feat
            if camids is None:
                raise ValueError('VTC requires camids')
            start = self.text_modality_index * per_modality
            rows = torch.arange(start, start + per_modality, device=x.device)
            cams = torch.cat([c.to(x.device) for c in camids], dim=0)[rows]
            feat_before = self.base(x[rows], trajectory_gate=torch.zeros(rows.shape[0], device=x.device))
            return cls_score, global_feat, feat, {
                'feat_after': global_feat[rows], 'feat_before': feat_before,
                'is_aerial': torch.isin(cams, self.aerial_cams.to(x.device)),
                'modality': torch.zeros_like(rows), 'n_view': self.prompts.n_view,
                'n_modality': self.prompts.n_modality, 'text': self.prompts(),
                'logit_scale': self.logit_scale, 'proj': self.base.clip_proj}
        global_feat = self.base(x)
        feat = self.bottleneck(global_feat)
        return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        state = torch.load(trained_path, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        # 八组实验不跨数据集迁移，必须完整恢复包括 BN、Trajectory 在内的参数。
        self.load_state_dict(state, strict=True)
        print('Loaded complete retrieval checkpoint: {}'.format(trained_path))


def make_model(cfg, num_class, camera_num=0, view_num=0):
    if cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_clip':
        raise ValueError('this experiment uses vit_base_clip')
    return build_transformer(num_class, cfg)
