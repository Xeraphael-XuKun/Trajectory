"""核对原P3初始化、KD/预测梯度边界及教师无关部署；不是正式训练。"""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from config import cfg
from model import make_model
from model.m2_teacher import normalized_clip_logits, kd_kl
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from tools.m2.artifacts import VERSION, seed_stage, validate_config, validate_text_bank
from utils.history_training import history_gradient_groups, clip_history_gradients
from utils.m5_checkpoint import (attach_metadata, load_teacher, restore_training,
                                 save_training, student_signature)


def optimizer_spec(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [(names[id(g['params'][0])], g['lr'], g['weight_decay'])
            for g in optimizer.param_groups if names[id(g['params'][0])] != 'base.clip_proj']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--student-only', action='store_true',
                        help='不读教师/文本产物，使用固定合成KD目标，仅验证学生接口')
    parser.add_argument('--output', required=True)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；不自动改用CPU')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    c = cfg.clone()
    c.merge_from_file(args.config_file)
    c.merge_from_list(args.opts)
    validate_config(c)
    assert c.HISTORY.METHOD == 'p3' and c.SOLVER.IMS_PER_BATCH == 64
    assert c.SOLVER.MAX_EPOCHS == 60 and c.HISTORY.LOSS_WEIGHT == 1.
    assert c.M2.KD_WEIGHT == .5 and c.M2.KD_TEMPERATURE == 2.
    assert c.M2.TEXT_TEMPERATURE == .07 and not c.M2.RESUME
    pure_cfg = c.clone()
    pure_cfg.M2.ENABLED = False
    seed_stage(c)
    pure = make_model(pure_cfg, 500, 7, 1).eval()
    pure_rng = torch.get_rng_state().clone()
    seed_stage(c)
    model = make_model(c, 500, 7, 1).eval()
    assert torch.equal(pure_rng, torch.get_rng_state())
    assert model.state_dict().keys() == pure.state_dict().keys()
    for key, value in pure.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    pure.to(device)
    model.to(device)
    center = torch.nn.Linear(1, 1)
    optimizer, _ = make_optimizer(c, model, center)
    pure_optimizer, _ = make_optimizer(pure_cfg, pure, center)
    assert optimizer_spec(model, optimizer) == optimizer_spec(pure, pure_optimizer)
    adapter = model.base.history_adapter
    assert model.base.token_trajectory is None
    assert sum(p.numel() for p in adapter.parameters()) == 2818112
    groups_by_id = {id(p): g for g in optimizer.param_groups for p in g['params']}
    for name, p in adapter.named_parameters():
        group = groups_by_id[id(p)]
        assert group['lr'] == (1e-4 if name.startswith('predictors.') else .00035)
        assert group['weight_decay'] == (0. if name == 'innovation_gain' else .0001)
    assert id(model.base.clip_proj) not in groups_by_id and not model.base.clip_proj.requires_grad
    probe = torch.linspace(-1, 1, 2 * 3 * 256 * 128, device=device).reshape(2, 3, 256, 128)
    with torch.no_grad():
        for gain in (0., .003):
            for m in (pure, model):
                m.base.history_adapter.velocity_gain.fill_(gain)
                m.base.history_adapter.innovation_gain.fill_(gain)
            for neck in ('after', 'before'):
                pure.neck_feat = model.neck_feat = neck
                torch.testing.assert_close(pure(probe, mode=1), model(probe, mode=1), rtol=0, atol=0)
        adapter.velocity_gain.zero_()
        adapter.innovation_gain.zero_()
    del pure, pure_optimizer
    gc.collect()
    model.train()
    teacher = None
    if args.student_only:
        text = F.normalize(torch.linspace(-1, 1, 500 * 512).reshape(500, 512).to(device), dim=-1)
        model.m5_metadata = dict(variant=c.M2.VARIANT, implementation=VERSION,
                                 stage='student', student=student_signature(model))
    else:
        bank = torch.load(c.M2.TEXT_BANK, map_location='cpu', weights_only=False)
        validate_text_bank(bank, c, 500)
        text = bank['text_bank'].float().to(device).detach()
        rng_before = torch.get_rng_state().clone()
        teacher, teacher_obj = load_teacher(c, 500, device)
        assert torch.equal(rng_before, torch.get_rng_state()), '教师加载消耗了学生RNG'
        assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
        assert teacher.base.history_adapter is None and teacher.base.token_trajectory is None
        attach_metadata(model, c, bank, teacher_obj)
        del teacher_obj, bank
        gc.collect()
    # 小batch仅用于数值和接口检查；生产预检另走原192图训练入口。
    joined = torch.linspace(-1, 1, 6 * 3 * 256 * 128, device=device).reshape(6, 3, 256, 128)
    images = list(joined.split(2))
    cameras = [torch.tensor([0, 5], device=device) for _ in range(3)]
    target = torch.tensor([0, 1], device=device).repeat(3)
    modalities = torch.arange(3, device=device).repeat_interleave(2)
    all_cameras = torch.cat(cameras)
    platforms = ((all_cameras == 5) | (all_cameras == 6)).long()
    assert modalities.tolist() == [0, 0, 1, 1, 2, 2]
    assert platforms.tolist() == [0, 1, 0, 1, 0, 1]
    before_bn = int(model.bottleneck.num_batches_tracked)
    score, global_feat, _, aux = model(images, camids=cameras, history_diagnostics=True)
    assert int(model.bottleneck.num_batches_tracked) == before_bn + 1
    with torch.no_grad():
        if teacher is None:
            lt = torch.linspace(-2, 2, 6 * 500, device=device).reshape(6, 500)
        else:
            gt = teacher.forward_pre_bn(joined, modalities, platforms)
            lt = normalized_clip_logits(gt, teacher.base.clip_proj, text, c.M2.TEXT_TEMPERATURE)
    ls = normalized_clip_logits(global_feat, model.base.clip_proj, text, c.M2.TEXT_TEMPERATURE)
    kd = kd_kl(lt, ls, c.M2.KD_TEMPERATURE)
    pred = history_prediction_loss(aux)
    assert torch.isfinite(kd) and kd > 0
    parameters = [adapter.velocity_gain, adapter.innovation_gain,
                  model.base.blocks[0].attn.qkv.weight, model.bottleneck.weight,
                  model.classifier.weight, adapter.predictors[0][-1].weight]
    kd_grads = torch.autograd.grad(kd, parameters, retain_graph=True, allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in kd_grads[:3])
    assert all(g is None for g in kd_grads[3:])
    pred_grads = torch.autograd.grad(pred, parameters, retain_graph=True, allow_unused=True)
    assert all(g is None for g in pred_grads[:-1]) and pred_grads[-1].norm() > 0
    torch.testing.assert_close(kd, 4 * F.kl_div((ls / 2).log_softmax(-1),
                               (lt / 2).softmax(-1), reduction='batchmean'), rtol=1e-4, atol=1e-6)
    ce = F.cross_entropy(score, target)
    triplet = TripletLoss()(global_feat, target)[0]
    total = ce + triplet + pred + .5 * kd
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.base.clip_proj.grad is None
    if teacher is not None:
        assert all(p.grad is None for p in teacher.parameters())
    clip_groups = history_gradient_groups(model)
    assert not ({id(p) for p in clip_groups['task']} & {id(p) for p in clip_groups['prediction']})
    clip_history_gradients(clip_groups, max_norm=1.)
    optimizer.step()
    assert adapter.velocity_gain.norm() > 0 and adapter.innovation_gain.norm() > 0
    losses = dict(ce=float(ce.detach()), triplet=float(triplet.detach()),
                  prediction=float(pred.detach()), kd=float(kd.detach()))
    del score, global_feat, aux, kd_grads, pred_grads, total, ce, triplet, kd, pred, ls, lt
    optimizer.zero_grad(set_to_none=True)
    del teacher
    gc.collect()
    model.eval()
    with tempfile.TemporaryDirectory(prefix='reload-', dir=output) as temporary:
        if args.student_only:
            checkpoint = Path(temporary) / 'student.pth'
            torch.save(dict(state_dict=model.state_dict(), metadata=model.m5_metadata), checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['state_dict'], strict=True)
        else:
            scheduler = create_scheduler(c, optimizer, iters_per_epoch=1)
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            save_training(temporary, 'transformer', 1, model, optimizer, scheduler, scaler, 1, 1, 0, 0.)
            saved_rng = torch.get_rng_state().clone()
            torch.rand(1)
            restored = restore_training(str(Path(temporary) / 'transformer_1.pth'),
                                        model, optimizer, scheduler, scaler)
            assert restored[:4] == (2, 1, 1, 0) and torch.equal(saved_rng, torch.get_rng_state())
            checkpoint = Path(temporary) / 'transformer_1_student.pth'
        deploy_cfg = c.clone()
        deploy_cfg.M2.ENABLED = False
        deploy_cfg.MODEL.PRETRAIN_CHOICE = 'no'
        deploy_cfg.M2.TEXT_BANK = deploy_cfg.M2.TEACHER_WEIGHT = 'unavailable-at-inference'
        deployment = make_model(deploy_cfg, 500, 7, 1).to(device).eval()
        deployment.load_param(str(checkpoint))
        with torch.no_grad():
            for neck in ('after', 'before'):
                model.neck_feat = deployment.neck_feat = neck
                torch.testing.assert_close(model(joined[:2], mode=1), deployment(joined[:2], mode=1), rtol=0, atol=0)
        deployment.base.history_adapter.specification[-1] = 1.
        try:
            deployment.load_param(str(checkpoint))
        except ValueError:
            pass
        else:
            raise AssertionError('错误的P3 alpha未被拒绝')
    report = dict(status='STUDENT_INTERFACE_ONLY_OK' if args.student_only else 'COMBINATION_BOUNDARIES_OK',
                  device=str(device), torch=torch.__version__, formal_training=False,
                  actual_original_teacher=not args.student_only, synthetic_images=6,
                  initialization_rng_state_equal=True, zero_nonzero_gain_pre_post_equal=True,
                  optimizer_equal_except_frozen_P0=True, separate_task_predictor_clip=True,
                  kd_to_backbone_and_both_gains=True, kd_excludes_predictor_bn_classifier=True,
                  prediction_only_to_predictors=True, strict_state_and_deployment_reload=True,
                  teacher_free_both_readouts_equal=True, wrong_alpha_rejected=True, losses=losses)
    (output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
