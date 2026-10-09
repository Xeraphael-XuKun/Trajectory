"""M2-3 identity prompt bank and shared deep text context."""
import os
import torch
import torch.nn as nn
from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer, SOT, EOT

class DeepIdentityText(nn.Module):
    """Frozen CLIP tower with four pre-ID slots and four identity slots."""
    def __init__(self, clip_state, num_classes, bank_path=None, n_ctx=4,
                 layers=11, context_length=77):
        super().__init__(); self.n_ctx, self.layers = int(n_ctx), int(layers)
        self.text = CLIPTextEncoder(); self.text.load_clip(clip_state); self.text.freeze()
        tok = build_tokenizer(); ph = tok.encode('X')[0]
        ids = [SOT] + tok.encode('a photo of a') + [ph] * 8 + tok.encode('person .') + [EOT]
        if len(ids) > context_length: raise ValueError('M2-3 template exceeds CLIP context length')
        self.register_buffer('token_ids', torch.tensor(ids + [0] * (context_length-len(ids)), dtype=torch.long))
        self.d_slots, self.s_slots = tuple(range(5,9)), tuple(range(9,13)); self.eot_index = len(ids)-1
        with torch.no_grad():
            emb = self.text.token_embedding(self.token_ids.unsqueeze(0)); hidden = self._hidden_states(emb)
            init = torch.stack([h[0, list(self.d_slots)] for h in hidden[1:]], 0)
        if init.shape[0] != 11: raise ValueError('M2-3 requires 11 deep text contexts')
        self.contexts = nn.Parameter(init.clone())
        self.register_buffer('id_bank', torch.zeros(int(num_classes), 4, 512)); self.register_buffer('text_anchors', torch.zeros(int(num_classes), 512))
        x = self.text.token_embedding.weight.detach()[ph]
        self.register_buffer('id_init', x[None,None,:].expand(int(num_classes),4,-1).clone(), persistent=False)
        if bank_path:
            if not os.path.exists(bank_path): raise FileNotFoundError('M2-3 TEXT_BANK missing: {}'.format(bank_path))
            bank = torch.load(bank_path, map_location='cpu', weights_only=False)
            if not hasattr(bank.get('id_tokens'), 'shape') or not hasattr(bank.get('anchors'), 'shape') or tuple(bank['id_tokens'].shape) != tuple(self.id_bank.shape) or tuple(bank['anchors'].shape) != tuple(self.text_anchors.shape):
                raise ValueError('M2-3 text bank shape mismatch')
            self.id_bank.copy_(bank['id_tokens']); self.text_anchors.copy_(bank['anchors'])
        else: self.id_bank.copy_(self.id_init)

    def _hidden_states(self, embeddings):
        x=(embeddings+self.text.positional_embedding.to(embeddings.dtype)).permute(1,0,2); out=[x.permute(1,0,2)]
        for blk in self.text.resblocks:
            x=blk(x,self.text.attn_mask.to(x.dtype)); out.append(x.permute(1,0,2))
        return out

    def forward(self, ids):
        ids=ids.long().to(self.token_ids.device); emb=self.text.token_embedding(self.token_ids.unsqueeze(0).expand(ids.numel(),-1)).clone()
        emb[:,self.s_slots]=self.id_bank.index_select(0,ids).to(emb.dtype)
        eot=torch.full((ids.numel(),),self.eot_index,dtype=torch.long,device=ids.device)
        return torch.nn.functional.normalize(self.text.forward_deep(emb,eot,self.d_slots,self.contexts).float(),dim=-1)
