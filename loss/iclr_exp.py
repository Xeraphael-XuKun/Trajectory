"""I1 reduction: sum of platform means, with both live visual branches."""
import torch
import torch.nn.functional as F


def tgva_logits(g_on, g_off, projection, text, scale):
    with torch.cuda.amp.autocast(enabled=False):
        on = F.normalize(g_on.float() @ projection.float(), dim=-1)
        off = F.normalize(g_off.float() @ projection.float(), dim=-1)
        return scale.float() * (on @ text.float().t()), scale.float() * (off @ text.float().t())


def tgva_loss(h_on, h_off, mods, views):
    with torch.cuda.amp.autocast(enabled=False):
        ce_on = F.cross_entropy(h_on.float(), 2 * mods, reduction='none')
        ce_off = F.cross_entropy(h_off.float(), 2 * mods + views, reduction='none')
        per_image = ce_on + ce_off
        raw = per_image.sum() * 0.0
        for v in (0, 1):
            selected = views == v
            if selected.any():
                raw = raw + per_image[selected].mean()
        return raw


def tgva_from_aux(aux):
    on, off = tgva_logits(aux['g_on'], aux['g_off'], aux['projection'],
                         aux['text'], aux['scale'])
    return tgva_loss(on, off, aux['mods'], aux['views']), on, off
