"""Frozen CLIP text with four ID, two modality and two platform tokens."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer, SOT, EOT
from utils.m1_artifacts import file_hash

def prompt_signature():
    tokenizer = build_tokenizer()
    placeholder = tokenizer.encode('X')
    tokens = [SOT] + tokenizer.encode('a photo of a X X X X X X X X person.') + [EOT]
    slots = [i for i, token in enumerate(tokens) if token == placeholder[0]]
    if len(placeholder) != 1 or len(slots) != 8:
        raise ValueError('M2-1 requires eight actual tokenizer slots')
    return {'template': 'a photo of a S S S S U U Q Q person.',
            'token_ids': tokens + [0] * (77 - len(tokens)), 'eot': len(tokens) - 1,
            'id_slots': slots[:4], 'modality_slots': slots[4:6], 'platform_slots': slots[6:],
            'tokenizer_sha256': file_hash(Path(__file__).resolve().parents[1] / 'backbones/bpe_simple_vocab_16e6.txt.gz')}

class ConditionPrompt(nn.Module):
    def __init__(self, clip_state, num_classes):
        super().__init__()
        self.text = CLIPTextEncoder()
        self.text.load_clip(clip_state)
        self.text.freeze().eval()
        self.signature = prompt_signature()
        self.register_buffer('token_ids', torch.tensor(self.signature['token_ids']))
        self.identity = nn.Parameter(torch.randn(num_classes, 4, 512) * .02)
        self.modality = nn.Parameter(torch.randn(3, 2, 512) * .02)
        self.platform = nn.Parameter(torch.randn(2, 2, 512) * .02)

    def train(self, mode=True):
        super().train(mode)
        self.text.eval()
        return self

    def forward(self, ids, modalities, platforms):
        with torch.cuda.amp.autocast(enabled=False):
            embeddings = self.text.token_embedding(self.token_ids[None].expand(len(ids), -1)).clone()
            embeddings[:, self.signature['id_slots']] = self.identity[ids]
            embeddings[:, self.signature['modality_slots']] = self.modality[modalities]
            embeddings[:, self.signature['platform_slots']] = self.platform[platforms]
            eot = torch.full_like(ids, self.signature['eot'])
            return F.normalize(self.text(embeddings.float(), eot).float(), dim=-1)
