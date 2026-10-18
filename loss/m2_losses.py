"""M2-5 FP32 distillation losses."""
from model.m2_teacher import kd_kl
def student_teacher_kd(student_logits, teacher_logits, temperature=2.0, weight=.5):
    return float(weight)*kd_kl(teacher_logits,student_logits,temperature)
