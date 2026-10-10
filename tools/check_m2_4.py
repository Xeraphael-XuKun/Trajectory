"""M2-4 必要检查：相机均衡/九组KL/梯度/原学生部署。"""
import argparse
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from config import cfg as defaults
from model import make_model
from loss.clip_relation import identity_modality_centers, relation_kl
from loss.triplet_loss import TripletLoss
from tools.m2_cache import load_relation_bank
from utils.m4_artifacts import train_manifest, camera_balanced_centers, PREPROCESS
from utils.m4_checkpoint import attach_metadata, save_training, restore_training
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

def reference_relation(centers, teacher, valid, ts, tt):
    terms = []
    for m in range(3):
        for n in range(3):
            for y in range(len(centers)):
                candidates = valid[:, n].clone()
                candidates[y] = False
                if not valid[y, m] or not candidates.any():
                    continue
                tlog = F.log_softmax(teacher[y, candidates].detach() / tt, 0)
                slog = F.log_softmax((centers[y, m] @ centers[candidates, n].t()) / ts, 0)
                terms.append((tlog.exp() * (tlog - slog)).sum())
    return torch.stack(terms).mean()

def check_formulas(device):
    # Unequal images per camera must not weight the first camera three times.
    features = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
    center, rows = camera_balanced_centers(features, torch.zeros(4,dtype=torch.long), torch.tensor([0,0,0,5]), 1)
    torch.testing.assert_close(center[0], F.normalize(torch.tensor([.5,.5]), dim=0))
    assert rows == [(0,0,3),(0,5,1)]
    features = torch.randn(36, 8, device=device)
    labels = torch.tensor([7,2,10], device=device).repeat_interleave(4).repeat(3)
    mods = torch.arange(3, device=device).repeat_interleave(12)
    actual, ids, valid = identity_modality_centers(features, labels, mods)
    expected = torch.stack([torch.stack([F.normalize(F.normalize(features[(labels==y)&(mods==m)],dim=1).mean(0),dim=0) for m in range(3)]) for y in ids])
    torch.testing.assert_close(actual, expected)
    order = torch.randperm(36, device=device)
    shuffled, shuffled_ids, _ = identity_modality_centers(features[order], labels[order], mods[order])
    torch.testing.assert_close(shuffled, actual)
    assert torch.equal(shuffled_ids, ids)
    centers = F.normalize(torch.randn(4,3,8,device=device),dim=-1).detach().requires_grad_()
    u = F.normalize(torch.randn(4,5,device=device),dim=-1)
    teacher = (u@u.t()).requires_grad_()
    full = torch.ones(4,3,dtype=torch.bool,device=device)
    ragged = torch.tensor([[1,1,0],[1,0,0],[1,1,1],[0,1,0]],dtype=torch.bool,device=device)
    for mask in (full, ragged):
        loss, stats = relation_kl(centers, teacher, .13, .09, mask, True)
        reference = reference_relation(centers, teacher, mask, .13, .09)
        torch.testing.assert_close(loss, reference)
        grad = torch.autograd.grad(loss, centers, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(reference, centers, retain_graph=True)[0]
        torch.testing.assert_close(grad, ref_grad, atol=2e-6, rtol=1e-4)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert torch.autograd.grad(loss, teacher, allow_unused=True, retain_graph=True)[0] is None
        assert stats['rows'].sum() > 0
    zero = relation_kl(centers[:2], teacher[:2,:2])
    assert zero == 0 and torch.isfinite(zero)
    assert torch.autograd.grad(zero, centers)[0].abs().sum() == 0
    print('相机等权、标签scatter/乱序、完整及缺失九组KL/两端梯度/两种温度、教师detach与小于3ID零项 PASS',flush=True)

def check(cfg, device='cpu', export_dir=None):
    torch.set_num_threads(2)
    device = torch.device(device)
    check_formulas(device)
    rows, mapping = train_manifest(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.SUBDIR)
    bank = load_relation_bank(cfg,len(mapping),device)
    torch.manual_seed(cfg.SOLVER.SEED)
    model = make_model(cfg,len(mapping),7,3).to(device)
    rng = torch.get_rng_state()
    initial = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    off = cfg.clone()
    off.M2.ENABLED = False
    off.M2.CLIP_PATH = off.M2.RELATION_BANK = off.M2.RGB_CENTERS = '/missing/artifact'
    torch.manual_seed(cfg.SOLVER.SEED)
    reference = make_model(off,len(mapping),7,3).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,reference.state_dict()[key].cpu()) for key,value in initial.items())
    assert all(key.startswith(('base.','bottleneck.','classifier.')) for key in model.state_dict())
    del initial
    transform = transforms.Compose([transforms.Resize((256,128)),transforms.ToTensor(),transforms.Normalize(PREPROCESS['mean'],PREPROCESS['std'])])
    sample_ids = [y for y in range(len(mapping)) if all(any(r[1]==y and r[3]==m for r in rows) for m in range(3))][:3]
    assert len(sample_ids) == 3
    images = []
    for m in range(3):
        current = []
        for y in sample_ids:
            path = next(r[0] for r in rows if r[1]==y and r[3]==m)
            with Image.open(path) as image:
                current.append(transform(image.convert('RGB')))
        images.append(torch.stack(current).to(device))
    model.eval(); reference.eval()
    with torch.no_grad():
        for readout in ('before','after'):
            model.neck_feat = reference.neck_feat = readout
            torch.testing.assert_close(model(images[0],mode=1),reference(images[0],mode=1),rtol=0,atol=0)
    del reference
    gc.collect()
    model.train()
    labels = torch.tensor(sample_ids,device=device)
    bn_before = model.bottleneck.num_batches_tracked.item()
    with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
        score,g,f = model(images,labels)
        centers,ids,valid = identity_modality_centers(f,labels.repeat(3),torch.arange(3,device=device).repeat_interleave(3))
        extra = cfg.M2.RELATION_WEIGHT * relation_kl(centers,bank['relation'][ids][:,ids],cfg.M2.STUDENT_TEMPERATURE,cfg.M2.TEACHER_TEMPERATURE,valid)
    assert model.bottleneck.num_batches_tracked.item() == bn_before + 1
    gradients = torch.autograd.grad(extra,[model.base.token_trajectory.gain,model.base.patch_embed.proj.weight,model.bottleneck.weight,model.classifier.weight],allow_unused=True,retain_graph=True)
    for grad in gradients[:3]:
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert gradients[3] is None
    optimizer,_ = make_optimizer(cfg,model,torch.nn.Linear(2,2))
    groups = {id(p):group for group in optimizer.param_groups for p in group['params']}
    assert len(groups) == sum(len(group['params']) for group in optimizer.param_groups)
    assert id(model.base.clip_proj) not in groups
    assert groups[id(model.base.token_trajectory.gain)]['lr'] == 3.5e-4
    assert groups[id(model.base.patch_embed.proj.weight)]['lr'] == 5e-6
    total = F.cross_entropy(score.float(),labels.repeat(3)) + TripletLoss()(g,labels.repeat(3))[0] + extra
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    assert model.base.token_trajectory.gain.abs().sum() > 0
    print('M2开关初始化/RNG/eval一致、单次BN、backbone/gain/BN梯度、classifier无辅助梯度及规定LR PASS',flush=True)
    if export_dir:
        Path(export_dir).mkdir(parents=True,exist_ok=True)
        attach_metadata(model,cfg,bank)
        scheduler = create_scheduler(cfg,optimizer,iters_per_epoch=1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        save_training(export_dir,'check',1,model,optimizer,scheduler,scaler,1,1,0)
        model.eval()
        features = {}
        with torch.no_grad():
            for readout in ('before','after'):
                model.neck_feat = readout
                features[readout] = model(images[0],mode=1).clone()
        assert restore_training(str(Path(export_dir)/'check_1.pth'),model,optimizer,scheduler,scaler)==(2,1,1,0)
        off.MODEL.PRETRAIN_CHOICE = 'no'
        student = make_model(off,len(mapping),7,3).to(device)
        student.load_param(str(Path(export_dir)/'check_1_student.pth'))
        student.eval()
        with torch.no_grad():
            for readout in ('before','after'):
                student.neck_feat = readout
                torch.testing.assert_close(student(images[0],mode=1),features[readout],rtol=0,atol=0)
        student.base.token_trajectory.acceleration_mix = 1.
        try:
            student.load_param(str(Path(export_dir)/'check_1_student.pth'))
        except ValueError:
            pass
        else:
            raise AssertionError('accepted wrong acceleration mix')
        print('非零gain完整保存/恢复、无教师工件原C0部署pre/post-BN完全相同、错误加速度设置拒绝 PASS',flush=True)
    print('M2-4必要方法检查通过；不是GPU成本或60epoch/检索收益验证。',flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--export-dir')
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = defaults.clone()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    check(cfg,args.device,args.export_dir)

if __name__=='__main__':
    main()
