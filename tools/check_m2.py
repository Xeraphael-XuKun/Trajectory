"""M2 numerical/gradient checks; optional image-batch CUDA preflight is separate."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from config import cfg as defaults
from model.m2 import M2Readout, frozen_bn_pair, validate_m2_config
from loss.m2_losses import m2_a_loss, m2_b_loss, m2_b_from_gaps


def reference_a(z1, z0, pid, cam, mod, cfg):
    s1, s0 = z1 @ z1.T, z0 @ z0.T
    groups = []
    for v in (0, 1):
        for m in range(3):
            anchors = []
            for i in range(len(pid)):
                gallery = [j for j in range(len(pid)) if cam[j] != cam[i]
                           and mod[j] == m and int(cam[j] in (5, 6)) == v]
                pos = [j for j in gallery if pid[j] == pid[i]]
                neg = [j for j in gallery if pid[j] != pid[i]]
                if not pos or not neg:
                    continue
                n = max(neg, key=lambda j: float(s1[i, j].detach()))
                d1 = s1[i, pos] - s1[i, n]
                d0 = s0[i, pos] - s0[i, n]
                term = cfg.TEMPERATURE * F.softplus((cfg.MARGIN-d1)/cfg.TEMPERATURE)
                if cfg.A_OBJECTIVE == "full":
                    term = term + (d0-d1-cfg.A_KEEP_TOL).relu()
                anchors.append(term.mean())
            if anchors:
                groups.append(torch.stack(anchors).mean())
    return torch.stack(groups).mean() if groups else z1.sum()*0


def reference_b(z, pid, cam, mod, cfg):
    s = F.normalize(z, dim=1) @ F.normalize(z, dim=1).T
    terms = []
    for i in range(len(pid)):
        for k in pid.unique():
            if k == pid[i]:
                continue
            for v in (0, 1):
                q = []
                for m in range(3):
                    gallery = [j for j in range(len(pid)) if cam[j] != cam[i]
                               and mod[j] == m and int(cam[j] in (5, 6)) == v]
                    pos = [j for j in gallery if pid[j] == pid[i]]
                    neg = [j for j in gallery if pid[j] == k]
                    if not pos or not neg:
                        break
                    def lme(rows):
                        values = s[i, rows] / cfg.TEMPERATURE
                        return cfg.TEMPERATURE * (values.logsumexp(0) -
                               values.new_tensor(float(len(rows))).log())
                    q.append(lme(neg)-lme(pos))
                if len(q) != 3:
                    continue
                q = torch.stack(q)
                a = torch.ones_like(q)
                if cfg.B_WEIGHTING == "conditional":
                    a = torch.stack([((q[m]-q[[n for n in range(3) if n != m]].max()
                          -cfg.B_DELTA).relu()/cfg.B_WEIGHT_SCALE).clamp(0,1)
                          for m in range(3)]).detach()
                terms.append((a*(q+cfg.MARGIN).relu().square()).mean())
    return torch.stack(terms).mean() if terms else z.sum()*0


def numerical_checks(device):
    torch.manual_seed(1234)
    # All six gallery conditions, deliberately unbalanced camera availability.
    rows = [(i, c, m) for i in range(3) for m in range(3) for c in (1, 2, 5, 6)]
    pid, cam, mod = torch.tensor(rows, device=device).T
    features = torch.randn(len(pid), 24, device=device, requires_grad=True)
    before = F.normalize(torch.randn_like(features), dim=1)
    conf = defaults.M2.clone()
    result = {}
    for mode in ("rank", "full"):
        conf.A_OBJECTIVE = mode
        z = F.normalize(features, dim=1)
        fast, stats = m2_a_loss(z, before, pid, cam, mod, conf)
        slow = reference_a(z, before, pid, cam, mod, conf)
        torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-6)
        g1 = torch.autograd.grad(fast, features, retain_graph=True)[0]
        g2 = torch.autograd.grad(slow, features, retain_graph=True)[0]
        torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-6)
        perm = torch.randperm(len(pid), device=device)
        other, _ = m2_a_loss(z[perm], before[perm], pid[perm], cam[perm], mod[perm], conf)
        torch.testing.assert_close(fast, other, rtol=1e-5, atol=1e-6)
        result["A_"+mode] = dict(loss=float(fast), **stats)
    for mode in ("uniform", "conditional"):
        conf.B_WEIGHTING = mode
        fast, stats = m2_b_loss(features, pid, cam, mod, conf)
        slow = reference_b(features, pid, cam, mod, conf)
        torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-6)
        g1 = torch.autograd.grad(fast, features, retain_graph=True)[0]
        g2 = torch.autograd.grad(slow, features, retain_graph=True)[0]
        torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-6)
        other, _ = m2_b_loss(features[perm], pid[perm], cam[perm], mod[perm], conf)
        torch.testing.assert_close(fast, other, rtol=1e-5, atol=1e-6)
        result["B_"+mode] = dict(loss=float(fast), **stats)
    assert result["B_uniform"]["valid_relations"] == result["B_conditional"]["valid_relations"]
    for mode in ("a", "b"):
        if mode == "a":
            loss, _ = m2_a_loss(F.normalize(features, dim=1), before, pid,
                               torch.ones_like(cam), mod, conf)
        else:
            loss, _ = m2_b_loss(features, pid, torch.ones_like(cam), mod, conf)
        grad = torch.autograd.grad(loss, features)[0]
        assert float(loss) == 0 and torch.isfinite(grad).all() and not grad.any()

    reader = M2Readout(24, 64, .1).to(device)
    g = torch.randn(36, 24, device=device, requires_grad=True)
    p = torch.randn(36, 8, 24, device=device, requires_grad=True)
    output, stats = reader(g, p)
    assert (output-g).abs().max() < 1e-6
    bn = nn.BatchNorm1d(24).to(device)
    bn_state = {k: v.clone() for k,v in bn.state_dict().items()}
    after, initial = frozen_bn_pair(reader, g, p, bn)
    conf.A_OBJECTIVE = "full"
    loss, _ = m2_a_loss(after, initial, pid, cam, mod, conf)
    loss.backward()
    assert g.grad is None and p.grad is None
    assert all(x.grad is None for x in bn.parameters())
    assert reader.scorer[-1].weight.grad.norm() > 0
    assert all(torch.equal(v, bn.state_dict()[k]) for k,v in bn_state.items())
    with torch.no_grad():
        reader.scorer[-1].weight.normal_(0,10)
        output, _ = reader(g, p)
        bound = float(((output-g).norm(dim=1)/g.norm(dim=1)).max())
        assert bound < .1
    # Identical patches and zero CLS must also have finite main-path gradients.
    reader.zero_grad()
    flat = torch.ones(2, 4, 24, device=device, requires_grad=True)
    zero = torch.zeros(2,24,device=device,requires_grad=True)
    output, _ = reader(zero,flat)
    output.sum().backward()
    assert torch.isfinite(flat.grad).all() and torch.isfinite(zero.grad).all()
    q = torch.tensor([[[[.05, -.3, -.4]]]], device=device, requires_grad=True)
    conf.B_WEIGHTING = "conditional"
    loss, _ = m2_b_from_gaps(q, torch.ones(1,1,1,device=device,dtype=torch.bool),conf)
    grad = torch.autograd.grad(loss,q)[0]
    assert grad.flatten()[0] > 0 and not grad.flatten()[1:].any()
    result["reader"] = dict(identity_and_detach="PASS", max_sampled_ratio=bound,
                            empty_relations="PASS", finite_degenerate_gradients="PASS")
    return result


def cached_features(path, device):
    batch = torch.load(path, map_location=device)
    tokens = batch["tokens"]
    reader = M2Readout().to(device)
    bn = nn.BatchNorm1d(768).to(device)
    bn.load_state_dict(batch["bn"])
    a,b = frozen_bn_pair(reader,tokens[:,0],tokens[:,1:],bn)
    conf = defaults.M2.clone()
    la,sa = m2_a_loss(a,b,batch["pid"],batch["cam"],batch["mod"],conf)
    la.backward()
    assert reader.scorer[-1].weight.grad.norm() > 0
    # Explicitly a batch-BN proxy over cached eval-backbone tokens, not a training replay.
    z = bn(tokens[:,0]).detach().requires_grad_(True)
    lb,sb = m2_b_loss(z,batch["pid"],batch["cam"],batch["mod"],conf)
    lb.backward()
    assert torch.isfinite(z.grad).all()
    return dict(scope="cached real PKM images, eval backbone; B batch-BN proxy",
                A=dict(loss=float(la),**sa),B=dict(loss=float(lb),**sb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--cached-features")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {"torch":torch.__version__, "device":args.device,
              "numerical":numerical_checks(args.device)}
    if args.cached_features:
        result["cached_features"] = cached_features(args.cached_features,args.device)
    expected = {
        "AB-A0": ("a", "none", "conditional", True),
        "AB-A1": ("a", "rank", "conditional", True),
        "AB-A2": ("a", "full", "conditional", True),
        "AB-A3": ("a", "full", "conditional", False),
        "AB-B0": ("b", "full", "uniform", True),
        "AB-B2": ("b", "full", "conditional", True),
        "AB-B3": ("b", "full", "conditional", False),
    }
    config_root = Path(__file__).resolve().parents[1] / "configs/m2_ab"
    assert {p.stem for p in config_root.glob("*.yml")} == set(expected)
    for path in sorted(config_root.glob("*.yml")):
        cfg = defaults.clone()
        cfg.merge_from_file(str(path))
        validate_m2_config(cfg)
        assert (cfg.M2.MODE, cfg.M2.A_OBJECTIVE, cfg.M2.B_WEIGHTING,
                cfg.MODEL.TOKEN_TRAJECTORY) == expected[path.stem]
        assert cfg.MODEL.TOKEN_TRAJECTORY_ACCEL_MIX == 0.0
        assert cfg.TEST.NECK_FEAT == "after" and cfg.SOLVER.MAX_EPOCHS == 60
    result["seven_configs"] = "PASS"
    text = json.dumps(result,indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text,encoding="utf-8")
