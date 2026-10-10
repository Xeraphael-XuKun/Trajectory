"""M2-3 frozen identity text; the controller solely owns shared contexts."""
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer, SOT, EOT
from utils.m3_artifacts import file_hash

class DeepIdentityText(nn.Module):
    def __init__(self, clip_state, num_classes, n_ctx=4, train_id_tokens=False):
        super().__init__()
        if n_ctx != 4:
            raise ValueError('M2-3 defines four D slots and four identity slots')
        self.text = CLIPTextEncoder()
        self.text.load_clip(clip_state)
        self.text.freeze().eval()
        tok = build_tokenizer()
        placeholder = tok.encode('X')
        if len(placeholder) != 1:
            raise ValueError('X must be one tokenizer token')
        # Use the complete real template, including tokenizer punctuation.
        ids = [SOT] + tok.encode('a photo of a X X X X X X X X person.') + [EOT]
        slots = [i for i, token in enumerate(ids) if token == placeholder[0]]
        if len(slots) != 8:
            raise ValueError('M2-3 template must expose eight slots')
        self.d_slots, self.s_slots = tuple(slots[:4]), tuple(slots[4:])
        if max(self.d_slots) >= min(self.s_slots):
            raise ValueError('all D slots must precede all identity slots')
        self.eot_index = len(ids) - 1
        self.register_buffer('token_ids', torch.tensor(ids + [0] * (77 - len(ids))).long())
        self.id_bank = nn.Parameter(torch.empty(num_classes, 4, 512), requires_grad=train_id_tokens)
        nn.init.normal_(self.id_bank, std=0.02)
        self.register_buffer('text_anchors', torch.zeros(num_classes, 512))
        with torch.no_grad():
            emb = self.text.token_embedding(self.token_ids[None])
            x = (emb + self.text.positional_embedding).permute(1, 0, 2)
            hidden = []
            for layer, block in enumerate(self.text.resblocks):
                if layer > 0:
                    hidden.append(x[list(self.d_slots), 0].clone())
                x = block(x, self.text.attn_mask)
        self.register_buffer('context_init', torch.stack(hidden), persistent=False)
        bpe = Path(__file__).resolve().parents[1] / 'backbones/bpe_simple_vocab_16e6.txt.gz'
        self.signature = {'template': 'a photo of a D D D D S S S S person.',
                          'token_ids': self.token_ids.tolist(), 'd_slots': list(self.d_slots),
                          's_slots': list(self.s_slots), 'eot': self.eot_index,
                          'context_length': 77, 'tokenizer_sha256': file_hash(bpe)}

    def train(self, mode=True):
        super().train(mode)
        self.text.eval()
        return self

    def load_bank(self, bank):
        with torch.no_grad():
            self.id_bank.copy_(bank['id_tokens'])
            self.text_anchors.copy_(bank['anchors'])
        self.id_bank.requires_grad_(False)
        torch.testing.assert_close(bank['context_init'].float(), self.context_init.cpu(), rtol=1e-4, atol=1e-5)

    def splice(self, ids):
        ids = ids.to(device=self.token_ids.device, dtype=torch.long)
        emb = self.text.token_embedding(self.token_ids[None].expand(ids.numel(), -1)).clone()
        emb[:, self.s_slots] = self.id_bank.index_select(0, ids)
        eot = torch.full((ids.numel(),), self.eot_index, dtype=torch.long, device=ids.device)
        return emb, eot

    def prompt_forward(self, ids):
        with torch.cuda.amp.autocast(enabled=False):
            emb, eot = self.splice(ids)
            return F.normalize(self.text(emb.float(), eot).float(), dim=-1)

    def forward(self, ids, contexts):
        with torch.cuda.amp.autocast(enabled=False):
            emb, eot = self.splice(ids)
            return F.normalize(self.text.forward_deep(emb.float(), eot, self.d_slots, contexts).float(), dim=-1)
