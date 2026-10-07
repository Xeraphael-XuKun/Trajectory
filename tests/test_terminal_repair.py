"""Independent scalar formula and stop-gradient checks for the pilot loss."""
import unittest
import torch
import torch.nn.functional as F
from loss.terminal_repair import terminal_repair


class TerminalRepairTests(unittest.TestCase):
    def test_scalar_formula_and_gradient(self):
        torch.manual_seed(9)
        raw=torch.randn(9,7,requires_grad=True)
        on=F.normalize(raw,dim=-1)
        off=F.normalize(torch.randn(9,7),dim=-1).requires_grad_()
        ids=torch.tensor([0,0,0,1,1,1,2,2,2])
        cams=torch.tensor([0,1,5]*3); mods=torch.tensor([0,1,2]*3)
        for mode in ('uniform','self','reference'):
            actual,stats=terminal_repair(off,on,ids,cams,mods,weighting=mode)
            terms=[]
            for i in range(9):
                positives=[j for j in range(9) if ids[j]==ids[i] and cams[j]!=cams[i]
                           and (mods[j]!=mods[i] or (cams[j]>=5)!=(cams[i]>=5))]
                negatives=[j for j in range(9) if ids[j]!=ids[i] and cams[j]!=cams[i]]
                h=[]; losses=[]
                for j in positives:
                    score=on.detach() if mode=='self' else off.detach()
                    h.append(torch.ones(()) if mode=='uniform' else torch.stack([
                        torch.sigmoid((score[i]@score[k]-score[i]@score[j])/.07) for k in negatives]).mean())
                    losses.append(torch.stack([F.softplus((on[i]@on[k]-on[i]@on[j])/.07)
                                               for k in negatives]).mean())
                weights=torch.stack(h); weights=weights/weights.sum()
                terms.append((weights*torch.stack(losses)).sum())
            expected=torch.stack(terms).mean()
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
            ga=torch.autograd.grad(actual,raw,retain_graph=True)[0]
            ge=torch.autograd.grad(expected,raw,retain_graph=True)[0]
            torch.testing.assert_close(ga,ge,rtol=2e-5,atol=2e-6)
            self.assertIsNone(torch.autograd.grad(actual,off,allow_unused=True,retain_graph=True)[0])
            self.assertEqual(stats['valid_anchors'],9)

    def test_no_positive(self):
        z=F.normalize(torch.randn(4,5),dim=-1).requires_grad_()
        loss,stats=terminal_repair(z.detach(),z,torch.arange(4),torch.arange(4),torch.arange(4))
        self.assertEqual(loss.item(),0); self.assertEqual(stats['valid_anchors'],0)
        loss.backward(); self.assertEqual(z.grad.abs().sum().item(),0)


if __name__=='__main__': unittest.main()
