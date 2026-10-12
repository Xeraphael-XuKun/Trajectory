"""I1 six learnable anchors using the original, frozen CLIP text tower."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones.clip_text import CLIPTextEncoder, ViewPrompts

TEMPLATE = 'a photo of a X view X person with X X X X .'


class TGVAPrompts(nn.Module):
    def __init__(self, clip_state):
        super().__init__()
        tower = CLIPTextEncoder()
        tower.load_clip(clip_state)
        self.prompts = ViewPrompts(tower, template=TEMPLATE, n_ctx=4,
            view_words=('ground', 'aerial'),
            modality_words=('rgb', 'infrared', 'thermal'))
        self.register_buffer('logit_scale', clip_state['logit_scale'].float().exp().clone())
        if not torch.isfinite(self.logit_scale).all():
            raise ValueError('original CLIP logit_scale is nonfinite')
        self.prompts.text.eval()

    def train(self, mode=True):
        super().train(mode)
        self.prompts.text.eval()
        return self

    def forward(self):
        # Frozen parameters, live embedding gradients. Never use no_grad here.
        with torch.cuda.amp.autocast(enabled=False):
            return F.normalize(self.prompts().float(), dim=-1)

    def metadata(self):
        p = self.prompts
        return dict(template=TEMPLATE, anchor_order=['RGB-G', 'RGB-A', 'IR-G',
            'IR-A', 'Thermal-G', 'Thermal-A'], token_ids=p.token_ids.tolist(),
            view_slot=p.view_slot, modality_slot=p.modality_slot,
            context_slots=p.ctx_slots, eot_index=p.eot_index,
            logit_scale=float(self.logit_scale), prompt_parameters=4608)
