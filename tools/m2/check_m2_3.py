"""M2-3 必要方法检查：初始化、共享、梯度、关闭路径和真实文本塔。"""
import argparse
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from config import cfg as defaults
from model import make_model
from model.backbones.token_trajectory import DeepTextC0Controller
from processor.processor import deep_text_identity_loss
from solver import make_optimizer
from loss.triplet_loss import TripletLoss
from utils.m3_artifacts import train_manifest, PREPROCESS

def controller_startup(context, device):
    controller = DeepTextC0Controller(12, 129).to(device)
    with torch.no_grad():
        controller.context.copy_(context)
    assert sum(p.numel() for p in controller.parameters()) == 186368
    velocity = torch.randn(2, 129, 768, device=device)
    gain = torch.nn.Parameter(torch.zeros(1, 129, 768, device=device))
    def backward():
        controller.zero_grad()
        gain.grad = None
        correction = gain * F.layer_norm(velocity, (768,))
        result = controller(1, velocity, correction)
        result.sum().backward()
        return correction, result
    correction, result = backward()
    assert torch.equal(correction, result)
    assert gain.grad.abs().sum() > 0
    assert controller.o.weight.grad.abs().sum() == 0
    assert controller.context.grad.abs().sum() == 0
    with torch.no_grad():
        gain.fill_(0.01)
    correction, result = backward()
    assert torch.equal(correction, result)
    assert controller.o.weight.grad.abs().sum() > 0
    assert controller.q.weight.grad.abs().sum() == 0
    with torch.no_grad():
        controller.o.weight.normal_(std=0.001)
    _, result = backward()
    for parameter in (controller.q.weight, controller.k.weight, controller.v.weight, controller.context):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    stats = controller.last_stats[-1]
    assert stats[2] >= 0.75 and stats[3] <= 1.25
    print('控制器 186368 参数、正乘子范围和三阶段视觉梯度启动 PASS', flush=True)

def check(cfg, device):
    torch.set_num_threads(2)
    device = torch.device(device)
    cfg = cfg.clone()
    cfg.freeze()
    _, pid_map = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)
    torch.manual_seed(cfg.SOLVER.SEED)
    model = make_model(cfg, len(pid_map), 7, 3).to(device)
    rng = torch.get_rng_state()
    visual_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    if key.startswith(('base.', 'bottleneck.', 'classifier.')) and 'deep_c0_controller' not in key}
    off = cfg.clone()
    off.defrost()
    off.M2.ENABLED = False
    off.M2.TEXT_BANK = off.M2.CLIP_PATH = '/missing/artifact'
    off.freeze()
    torch.manual_seed(cfg.SOLVER.SEED)
    reference = make_model(off, len(pid_map), 7, 3).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, reference.state_dict()[key].cpu()) for key, value in visual_state.items())
    assert reference.base.deep_c0_controller is None and reference.deep_identity is None
    del visual_state
    rows, _ = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)
    transform = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                                     transforms.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])
    images = []
    for modality in range(3):
        current = []
        for pid in range(2):
            path = next(row[0] for row in rows if row[1] == pid and row[3] == modality)
            with Image.open(path) as image:
                current.append(transform(image.convert('RGB')))
        images.append(torch.stack(current).to(device))
    model.eval()
    reference.eval()
    controller = model.base.deep_c0_controller
    assert sum(p.numel() for p in controller.parameters()) == 186368
    assert all(p.bias is None for p in (controller.q, controller.k, controller.v, controller.o))
    context_keys = [key for key, parameter in model.named_parameters() if parameter is controller.context]
    assert context_keys == ['base.deep_c0_controller.context']
    with torch.no_grad():
        for readout in ('before', 'after'):
            model.neck_feat = reference.neck_feat = readout
            torch.testing.assert_close(model(images[0], mode=1), reference(images[0], mode=1), rtol=0, atol=0)
        ids = torch.arange(2, device=device)
        current = model.deep_identity(ids, controller.context)
        anchors = model.deep_identity.text_anchors[ids]
        torch.testing.assert_close(current, anchors, rtol=1e-4, atol=1e-5)
        assert model.deep_identity.signature['eot'] == model.deep_identity.eot_index
        assert max(model.deep_identity.d_slots) < min(model.deep_identity.s_slots)
    print('真实文本塔 t(C0)=t0、单一共享参数、初始化/RNG/关闭路径 PASS', flush=True)
    del reference
    gc.collect()
    controller_startup(controller.context.detach(), device)
    model.train()
    assert not model.deep_identity.text.training
    labels = torch.tensor([0, 1], device=device)
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        score, g, f, aux = model(images, labels)
    assert aux['text'].shape == (2, 512) and aux['visual'].shape == (6, 512)
    assert torch.equal(aux['targets'], labels.repeat(3))
    assert aux['text'].dtype == torch.float32 and aux['visual'].dtype == torch.float32
    extra, stats = deep_text_identity_loss(aux)
    text_grad = torch.autograd.grad(extra, controller.context, retain_graph=True)[0]
    assert torch.isfinite(text_grad).all() and text_grad.abs().sum() > 0
    assert all(value is None for value in torch.autograd.grad(extra,
               [model.bottleneck.weight, model.classifier.weight], retain_graph=True, allow_unused=True))
    assert not model.deep_identity.id_bank.requires_grad
    assert all(not p.requires_grad for p in model.deep_identity.text.parameters())
    assert not model.base.clip_proj.requires_grad
    # Formula and anchor-only control paths must retain the intended reductions.
    expected_it = F.cross_entropy(aux['visual'] @ aux['text'].t() / 0.07, aux['targets'])
    expected_anchor = (1 - (aux['text'] * aux['anchor']).sum(-1)).mean()
    torch.testing.assert_close(stats['loss_it'], expected_it.detach())
    torch.testing.assert_close(stats['loss_anchor'], expected_anchor.detach(), atol=2e-7, rtol=1e-4)
    optimizer, _ = make_optimizer(cfg, model, torch.nn.Linear(2, 2))
    groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
    for parameter in controller.parameters():
        assert groups[id(parameter)]['lr'] == 3.5e-4 and groups[id(parameter)]['weight_decay'] == 1e-4
    assert groups[id(model.base.token_trajectory.gain)]['lr'] == 3.5e-4
    assert groups[id(model.base.patch_embed.proj.weight)]['lr'] == 5e-6
    assert id(model.base.clip_proj) not in groups and id(model.deep_identity.id_bank) not in groups
    assert len(groups) == sum(len(group['params']) for group in optimizer.param_groups)
    total = F.cross_entropy(score.float(), labels.repeat(3)) + TripletLoss()(g, labels.repeat(3))[0] + extra
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    assert model.deep_identity.id_bank.grad is None and model.base.clip_proj.grad is None
    assert all(p.grad is None for p in model.deep_identity.text.parameters())
    # Export nontrivial modulation rather than testing only WO=0, G=0.
    with torch.no_grad():
        model.base.token_trajectory.gain.normal_(std=0.001)
        controller.o.weight.normal_(std=0.001)
    print('图文两侧上下文梯度、冻结/损失归约/唯一 optimizer 参数与 LR PASS', flush=True)
    return model, images[0], optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = defaults.clone()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    check(cfg, args.device)
    print('M2-3 小批量方法检查通过；这不是正式训练或检索增益验证。')

if __name__ == '__main__':
    main()
