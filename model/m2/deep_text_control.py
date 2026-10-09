"""M2-3 identity prompt bank and shared deep text context."""
import os
import torch
import torch.nn as nn
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer, SOT, EOT

class DeepIdentityText(nn.Module):
    """Frozen CLIP tower with four pre-ID slots and four identity slots."""
    def __init__(self, clip_state, num_classes, bank_path=None, n_ctx=4,
                 layers=11, context_length=77, train_id_tokens=False):
        super().__init__(); self.n_ctx, self.layers = int(n_ctx), int(layers)
        self.text = CLIPTextEncoder(); self.text.load_clip(clip_state); self.text.freeze()
        tok = build_tokenizer(); placeholder = tok.encode('X')
        if len(placeholder) != 1:
            raise ValueError('M2-3 X placeholder must be one tokenizer token')
        ph = placeholder[0]; prefix = [SOT] + tok.encode('a photo of a')
        ids = prefix + [ph] * (2 * self.n_ctx) + tok.encode('person .') + [EOT]
        if len(ids) > context_length: raise ValueError('M2-3 template exceeds CLIP context length')
        self.register_buffer('token_ids', torch.tensor(ids + [0] * (context_length-len(ids)), dtype=torch.long))
        self.d_slots = tuple(range(len(prefix), len(prefix) + self.n_ctx))
        self.s_slots = tuple(range(len(prefix) + self.n_ctx, len(prefix) + 2*self.n_ctx))
        self.eot_index = len(ids)-1
        with torch.no_grad():
            emb = self.text.token_embedding(self.token_ids.unsqueeze(0)); hidden = self._hidden_states(emb)
            # C_l is the input to text block l (l=2..12), i.e. Z_1..Z_11.
            # hidden[12] is Z_12, after the final block, and must not be used.
            init = torch.stack([h[0, list(self.d_slots)] for h in hidden[1:12]], 0)
        if init.shape[0] != 11: raise ValueError('M2-3 requires 11 deep text contexts')
        self.contexts = nn.Parameter(init.clone())
        self.id_bank = nn.Parameter(
            torch.zeros(int(num_classes), self.n_ctx, 512),
            requires_grad=bool(train_id_tokens))
        self.register_buffer('text_anchors', torch.zeros(int(num_classes), 512))
        x = self.text.token_embedding.weight.detach()[ph]
        self.register_buffer('id_init', x[None,None,:].expand(int(num_classes),self.n_ctx,-1).clone(), persistent=False)
        if bank_path:
            if not os.path.exists(bank_path): raise FileNotFoundError('M2-3 TEXT_BANK missing: {}'.format(bank_path))
            bank = torch.load(bank_path, map_location='cpu', weights_only=False)
            if not hasattr(bank.get('id_tokens'), 'shape') or not hasattr(bank.get('anchors'), 'shape') or tuple(bank['id_tokens'].shape) != tuple(self.id_bank.shape) or tuple(bank['anchors'].shape) != tuple(self.text_anchors.shape):
                raise ValueError('M2-3 text bank shape mismatch')
            with torch.no_grad():
                self.id_bank.copy_(bank['id_tokens'])
                self.text_anchors.copy_(bank['anchors'])
        else: self.id_bank.data.copy_(self.id_init)

    def _hidden_states(self, embeddings):
        x=(embeddings+self.text.positional_embedding.to(embeddings.dtype)).permute(1,0,2); out=[x.permute(1,0,2)]
        for blk in self.text.resblocks:
            x=blk(x,self.text.attn_mask.to(x.dtype)); out.append(x.permute(1,0,2))
        return out

    def prompt_forward(self, ids):
        """Stage-A P-ID encoding with fixed D=X slots and no deep replacement."""
        ids = ids.long().to(self.token_ids.device)
        emb = self.text.token_embedding(
            self.token_ids.unsqueeze(0).expand(ids.numel(), -1)).clone()
        emb[:, self.s_slots] = self.id_bank.index_select(0, ids).to(emb.dtype)
        eot = torch.full((ids.numel(),), self.eot_index, dtype=torch.long,
                         device=ids.device)
        return torch.nn.functional.normalize(self.text(emb, eot).float(), dim=-1)

    def forward(self, ids):
        ids=ids.long().to(self.token_ids.device); emb=self.text.token_embedding(self.token_ids.unsqueeze(0).expand(ids.numel(),-1)).clone()
        emb[:,self.s_slots]=self.id_bank.index_select(0,ids).to(emb.dtype)
        eot=torch.full((ids.numel(),),self.eot_index,dtype=torch.long,device=ids.device)
        return torch.nn.functional.normalize(self.text.forward_deep(emb,eot,self.d_slots,self.contexts).float(),dim=-1)
