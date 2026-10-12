"""Regression: reused teacher must isolate the student's R2 configuration."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from config import cfg
from model.m2_teacher import ConditionalVisualTeacher
from utils.r2_experiment import r2_signature


def main():
    c=cfg.clone()
    c.merge_from_file(str(Path(__file__).resolve().parents[2]/'configs/r2_m5_one_sided.yml'))
    c.freeze()
    before=c.dump()
    assert r2_signature(c)['kd_mode']=='one_sided_target'
    disabled=c.clone()
    disabled.defrost()
    disabled.R2.ENABLED=False
    disabled.freeze()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        reference=ConditionalVisualTeacher(disabled,500,load_original=False)
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(1234)
        actual=ConditionalVisualTeacher(c,500,load_original=False)
        assert torch.equal(torch.get_rng_state(),expected_rng)
        assert actual.base.token_trajectory is None
        assert list(actual.state_dict())==list(reference.state_dict())
        assert all(torch.equal(value,reference.state_dict()[name]) for name,value in actual.state_dict().items())
    assert c.is_frozen() and c.dump()==before and c.R2.ENABLED and c.M2.ENABLED
    assert r2_signature(c)['kd_mode']=='one_sided_target'
    print('TEACHER_R2_CONFIG_ISOLATION_STATE_AND_RNG_PASS',flush=True)


if __name__=='__main__':
    main()
