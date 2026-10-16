"""P3-B numerical formula, denominator gradients, P3 regression and real updates."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from config import cfg
from model import make_model
from model.backbones.history_p3_b import HistoryP3BAdapter
from loss.history_prediction import history_prediction_loss
from loss.triplet_loss import TripletLoss
from solver import make_optimizer
from utils.history_training import history_gradient_groups, clip_history_gradients
from check_history import historical_module
from check_history_p3 import seed, norm, formula_check, check_actual_velocity, legacy_check


def normalization_check():
    seed(1235)
    a = HistoryP3BAdapter(16,4,4,hidden_dim=3)
    assert a.tau == .25 and a.epsilon == 1e-5
    assert not hasattr(a,'history_retention') and not hasattr(a,'modal_embeddings')
    # Distinct scales per image/patch ensure neither variance is a batch statistic or constant.
    e = (torch.randn(2,4,16,dtype=torch.float64) * torch.tensor([1e-4,.1,1.,3.],dtype=torch.float64)[None,:,None] + .4).requires_grad_()
    z = (torch.randn_like(e) * torch.tensor([.2,.5,2.,4.],dtype=torch.float64)[None,:,None]).requires_grad_()
    direction = a._normalize_innovation(e,z)
    center = e - e.mean(-1,keepdim=True)
    ve = center.square().mean(-1,keepdim=True)
    vz = ((z-z.mean(-1,keepdim=True)).square().mean(-1,keepdim=True)).detach()
    expected = center / (ve + .0625*vz + 1e-5).sqrt()
    assert torch.allclose(direction,expected,atol=1e-12,rtol=1e-12)
    expected_rms = (ve/(ve+.0625*vz+1e-5)).sqrt().squeeze(-1)
    assert torch.allclose(direction.square().mean(-1).sqrt(),expected_rms,atol=1e-12,rtol=1e-12)
    assert torch.all(direction.square().mean(-1).sqrt() < 1)
    weights = torch.randn_like(direction)
    ge,gz = torch.autograd.grad((direction*weights).sum(),(e,z),allow_unused=True)
    manual_ge = torch.autograd.grad((expected*weights).sum(),e,retain_graph=True)[0]
    assert gz is None and torch.allclose(ge,manual_ge,atol=1e-11,rtol=1e-11)
    # Detaching innovation variance/mean would silently change this method's task derivative.
    wrong_var = center/(ve.detach()+.0625*vz+1e-5).sqrt()
    wrong_mean = (e-e.mean(-1,keepdim=True).detach())/(ve+.0625*vz+1e-5).sqrt()
    assert not torch.allclose(ge,torch.autograd.grad((wrong_var*weights).sum(),e,retain_graph=True)[0])
    assert not torch.allclose(ge,torch.autograd.grad((wrong_mean*weights).sum(),e)[0])
    for value in (0.,7.):
        constant = torch.full((2,4,16),value)
        assert torch.count_nonzero(a._normalize_innovation(constant,torch.zeros_like(constant))) == 0
    a.tau = 0.
    assert torch.allclose(a._normalize_innovation(e.detach().float(),z.detach().float()),
                          F.layer_norm(e.detach().float(),(16,),eps=1e-5),atol=2e-5,rtol=2e-5)
    a.tau = .25
    # End-to-end adapter formula and the original task/prediction gradient boundary.
    h,v1 = torch.randn(2,5,16),torch.randn(2,5,16,requires_grad=True)
    r1,hist,_,_ = a.correction(h,0,[],None,True)
    assert not hist and torch.count_nonzero(r1)==0
    _,hist2,_,item2 = a.correction(h,1,hist,None,True,previous_velocity=v1)
    assert item2['prediction'] is None and not hist2[-1].requires_grad
    v2 = torch.randn_like(v1,requires_grad=True)
    with torch.no_grad():
        a.velocity_gain.fill_(.02); a.innovation_gain.fill_(.03)
        a.predictors[0][-1].weight.normal_(std=.02)
    r3,hist3,_,item3 = a.correction(h,2,hist2,None,True,True,previous_velocity=v2)
    current = F.layer_norm(v2,(16,),eps=1e-5)
    delta = a.predictors[0](torch.cat((hist2[-1],torch.zeros_like(hist2[-1])),-1))
    innovation = current[:,1:]-(hist2[-1]+delta).detach()
    centered = innovation-innovation.mean(-1,keepdim=True)
    denominator = (centered.square().mean(-1,keepdim=True)+.0625*current[:,1:].var(-1,unbiased=False,keepdim=True).detach()+1e-5).sqrt()
    rv = a.velocity_gain[1]*current
    re = .25*a.innovation_gain[0]*centered/denominator
    assert torch.allclose(r3,rv+torch.cat((torch.zeros_like(rv[:,:1]),re),1),atol=1e-8,rtol=1e-6)
    assert torch.equal(r3[:,:1],rv[:,:1]) and torch.equal(item3['prediction'],delta)
    (r3*torch.randn_like(r3)).sum().backward()
    assert v2.grad is not None and v1.grad is None and norm(a.predictors.parameters()) == 0
    assert norm([a.velocity_gain]) > 0 and norm([a.innovation_gain]) > 0
    a.zero_grad(set_to_none=True); v2.grad = None
    history_prediction_loss({'layers':{3:item3},'prediction_layers':[3]}).backward()
    assert norm(a.predictors.parameters())>0 and a.velocity_gain.grad is None and a.innovation_gain.grad is None
    assert v1.grad is None and v2.grad is None
    assert len(hist3)==2 and all(not x.requires_grad for x in hist3)
    # Identical zero writes do not imply identical innovation-gain startup derivatives.
    original = historical_module('model/backbones/history_p3.py','model.backbones.p3b_unit_reference','2c71241').HistoryP3Adapter
    seed(); old = original(16,4,4,hidden_dim=3)
    seed(); new = HistoryP3BAdapter(16,4,4,hidden_dim=3)
    probe = torch.randn_like(h)
    ro = old.correction(h,2,hist2,None,previous_velocity=v2)[0]
    rn = new.correction(h,2,hist2,None,previous_velocity=v2)[0]
    assert torch.equal(ro,rn)
    (ro*probe).sum().backward(); (rn*probe).sum().backward()
    assert torch.equal(old.velocity_gain.grad,new.velocity_gain.grad)
    assert not torch.allclose(old.innovation_gain.grad,new.innovation_gain.grad)


def original_p3_regression(c,images,report):
    pc = c.clone(); pc.HISTORY.METHOD = 'p3'
    old_model = historical_module('model/make_model.py','model.p3b_reference_model','2c71241')
    old_adapter = historical_module('model/backbones/history_p3.py','model.backbones.p3b_reference_adapter','2c71241')
    seed(); reference = old_model.make_model(pc,500,7,2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(pc.HISTORY.INIT_SEED)
        reference.base.history_adapter = old_adapter.HistoryP3Adapter(768,12,128,32,.25,list(range(3,13)))
    rng = torch.random.get_rng_state().clone()
    seed(); candidate = make_model(c,500,7,2)
    assert torch.equal(rng,torch.random.get_rng_state())
    for k,v in reference.state_dict().items():
        if k != 'base.history_adapter.specification': assert torch.equal(v,candidate.state_dict()[k]),k
    assert dict(reference.named_parameters()).keys() == dict(candidate.named_parameters()).keys()
    with torch.no_grad():
        reference.base.history_adapter.velocity_gain.fill_(.002)
        reference.base.history_adapter.innovation_gain.fill_(.003)
        for p in reference.base.history_adapter.predictors:
            p[-1].weight.normal_(std=.002); p[-1].bias.normal_(std=.01)
    seed(); current = make_model(pc,500,7,2)
    current.load_state_dict(reference.state_dict(),strict=True)
    with torch.no_grad():
        seed(); left = reference(images,history_diagnostics=True)
        seed(); right = current(images,history_diagnostics=True)
    assert all(torch.equal(x,y) for x,y in zip(left[:3],right[:3]))
    assert torch.equal(history_prediction_loss(left[3]),history_prediction_loss(right[3]))
    old_solver = historical_module('solver/make_optimizer.py','solver.p3b_reference_solver','2c71241')
    old_opt,_ = old_solver.make_optimizer(pc,reference,nn.Linear(1,1))
    current_opt,_ = make_optimizer(c,candidate,nn.Linear(1,1))
    assert [(g['lr'],g['weight_decay']) for g in old_opt.param_groups] == [(g['lr'],g['weight_decay']) for g in current_opt.param_groups]
    with torch.no_grad():
        for k,v in reference.state_dict().items():
            if k != 'base.history_adapter.specification': candidate.state_dict()[k].copy_(v)
        candidate.base.history_adapter.tau = 0.
        reference.eval(); candidate.eval()
        left = reference(images,history_diagnostics=True)
        right = candidate(images,history_diagnostics=True)
    max_error = max(float((x-y).abs().max()) for x,y in zip(left[:3],right[:3]))
    assert all(torch.allclose(x,y,atol=2e-5,rtol=2e-5) for x,y in zip(left[:3],right[:3]))
    assert torch.allclose(history_prediction_loss(left[3]),history_prediction_loss(right[3]),atol=2e-6,rtol=2e-6)
    report.update(original_p3_source_forward_loss_optimizer_unchanged=True,
                  shared_parameters_and_rng_equal_p3=True,tau_zero_full_forward_max_abs_error=max_error,
                  tau_zero_full_forward_and_loss_close_p3=True)
    del reference,candidate,current,left,right,old_opt,current_opt; gc.collect()


def run(args):
    torch.set_num_threads(4)
    formula_check(); normalization_check()
    report = {'status':'LOCAL_CHECK_ONLY','torch':torch.__version__,'source_base':'2c71241',
              'formula_rms_and_denominator_gradient_boundary':True,'zero_gain_startup_derivative_changes':True,
              'formal_training':False}
    c = cfg.clone(); c.merge_from_file(str(ROOT/'configs/history_P3_B.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'imagenet' if args.pretrained else 'no'
    if args.pretrained: c.MODEL.PRETRAIN_PATH = args.pretrained
    images = [torch.randn(2,3,256,128) for _ in range(3)]
    original_p3_regression(c,images,report)
    bc = c.clone(); bc.HISTORY.ENABLED = False
    seed(); baseline = make_model(bc,500,7,2)
    rng = torch.random.get_rng_state().clone()
    seed(); model = make_model(c,500,7,2)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
    with torch.no_grad():
        seed(); left = baseline(images)
        seed(); right = model(images,history_diagnostics=True)
    assert all(torch.equal(x,y) for x,y in zip(left,right[:3]))
    assert right[3]['prediction_layers'] == list(range(3,13))
    assert all(right[3]['layers'][l]['prediction'] is None for l in (1,2))
    expected = torch.stack([F.smooth_l1_loss(right[3]['layers'][l]['prediction'],right[3]['layers'][l]['target']) for l in range(3,13)]).mean()
    assert torch.equal(expected,history_prediction_loss(right[3]))
    report['zero_gain_baseline_rng_and_ten_layer_loss'] = True
    del baseline,left,right; gc.collect()
    check_actual_velocity(model,images)
    report['actual_block_output_minus_corrected_input'] = True
    adapter = model.base.history_adapter
    assert sum(p.numel() for p in adapter.parameters()) == 2818112 and adapter.tau == .25
    if args.real_data:
        from datasets import make_dataloader
        dc = c.clone(); dc.DATASETS.ROOT_DIR = args.real_data
        dc.DATALOADER.NUM_WORKERS = 0; dc.SOLVER.IMS_PER_BATCH = 4; dc.DATALOADER.NUM_INSTANCE = 2
        seed(); loader,_,_,_,classes,_,_ = make_dataloader(dc)
        images,labels,_ = next(iter(loader)); assert classes==500
        report['real_batch'] = {'actual_images':sum(len(x) for x in images),'identities':labels.unique().numel()}
    else: labels = torch.tensor([0,1])
    target = labels.repeat(3)
    optimizer,_ = make_optimizer(c,model,nn.Linear(1,1))
    by_id = {id(p):g for g in optimizer.param_groups for p in g['params']}
    for name,p in adapter.named_parameters():
        family=name.split('.')[0]; g=by_id[id(p)]
        assert g['lr'] == (1e-4 if family=='predictors' else 3.5e-4)
        assert g['weight_decay'] == (0 if family=='innovation_gain' else 1e-4)
    groups = history_gradient_groups(model)
    assert not ({id(p) for p in groups['task']} & {id(p) for p in groups['prediction']})
    assert sum(len(v) for v in groups.values()) == sum(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in groups['prediction']) == 745280
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        seed(); out = model(images,history_diagnostics=True)
        task=F.cross_entropy(out[0],target)+TripletLoss()(out[1],target)[0]
        prediction=history_prediction_loss(out[3])
        task.backward(retain_graph=True)
        assert norm(adapter.predictors.parameters())==0
        assert norm([adapter.velocity_gain])>0 and norm([adapter.innovation_gain])>0
        task_grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        prediction.backward()
        assert norm(adapter.predictors.parameters())>0
        assert all(torch.equal(task_grads[n],p.grad) for n,p in model.named_parameters() if n in task_grads)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        clip_history_gradients(groups); optimizer.step()
    report['lr_wd_separate_clip_and_real_updates']=True
    report['adapter_parameters']=sum(p.numel() for p in adapter.parameters())
    model.zero_grad(set_to_none=True)
    del optimizer,out,task,prediction,groups,task_grads; gc.collect()
    model.eval()
    with torch.no_grad():
        first=model(images[0],mode=1); model(images[1],mode=2)
        assert torch.equal(first,model(images[0],mode=1))
        model.neck_feat='before'; pre=model(images[0],mode=1)
        assert torch.equal(pre,model.base(images[0])); model.neck_feat='after'
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'checkpoint.pth'; torch.save(model.state_dict(),checkpoint)
    with torch.no_grad(): adapter.innovation_gain.add_(1)
    model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'),strict=True)
    model.load_param(str(checkpoint))
    with torch.no_grad(): assert torch.equal(first,model(images[0],mode=1))
    assert not torch.equal(first,pre)
    spec=adapter.specification.clone()
    adapter.specification[-2]=0.
    try: model.load_param(str(checkpoint))
    except ValueError: report['wrong_tau_specification_rejected']=True
    else: raise AssertionError('B tau specification was not checked')
    adapter.specification.copy_(spec)
    report['strict_reload_readouts_and_state_reset']=True
    report['legacy_hi1_unchanged']=legacy_check(args)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('P3_B_CHECK_OK '+json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pretrained',default='')
    p.add_argument('--real-data',default='')
    p.add_argument('--output',default=str(ROOT/'work/p3_b_check'))
    run(p.parse_args())
