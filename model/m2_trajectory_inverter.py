"""M2-2 trajectory summary to four continuous CLIP pseudo-words."""
import math
import torch
from torch import nn
from .backbones.clip_text import SOT, EOT, PLACEHOLDER, build_tokenizer


class TrajectoryInverter(nn.Module):
    def __init__(self, text_encoder, width=128, n_words=4):
        super().__init__()
        if n_words != 4:
            raise ValueError('M2-2 defines exactly four pseudo-words')
        self.text = text_encoder
        self.n_words = n_words
        self.in_proj = nn.Linear(2304, width)
        self.norm = nn.LayerNorm(width)
        self.query = nn.Parameter(torch.randn(n_words, width) * 0.02)
        self.word = nn.Linear(width, 512)
        nn.init.normal_(self.word.weight, std=0.001)
        nn.init.zeros_(self.word.bias)
        # Fixed sinusoidal depth code, avoiding learned layer identity.
        pos = torch.arange(11, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / width))
        pe = torch.zeros(11, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('layer_code', pe, persistent=False)
        tok = build_tokenizer()
        ids = [SOT] + tok.encode('a photo of a X X X X person.') + [EOT]
        slots = [i for i, t in enumerate(ids) if t == tok.encode(PLACEHOLDER)[0]]
        if len(slots) != 4:
            raise RuntimeError('M2-2 template must expose four placeholder slots')
        self.register_buffer('token_ids', torch.tensor(ids + [0] * (text_encoder.context_length-len(ids))).long(), persistent=False)
        self.register_buffer('slot_ids', torch.tensor(slots).long(), persistent=False)
        self.eot_index = len(ids) - 1
        with torch.no_grad():
            self.x_embedding = text_encoder.token_embedding.weight[tok.encode(PLACEHOLDER)[0]].detach().clone()

    def forward(self, summary):
        # [N,11,2304] -> [N,4,512], with attention over depth only.
        h = self.norm(torch.nn.functional.gelu(self.in_proj(summary)) + self.layer_code[None])
        attn = torch.softmax(torch.einsum('kw,nlw->nkl', self.query, h) / math.sqrt(h.shape[-1]), dim=-1)
        pooled = torch.einsum('nkl,nlw->nkw', attn, h)
        words = self.x_embedding[None, None] + self.word(pooled)
        return words, attn

    def splice(self, words):
        ids = self.token_ids[None].expand(words.shape[0], -1)
        emb = self.text.token_embedding(ids).clone()
        emb[:, self.slot_ids, :] = words
        return emb, self.eot_index

