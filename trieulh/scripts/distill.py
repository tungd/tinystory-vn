"""Token-level knowledge-distillation loss (Hinton et al., 2015).

Student learns the teacher's softened next-token distribution (KL on logits)
plus the hard-label cross-entropy. Used to distill the 30M teacher into a 10M
student on story tokens only (conditioning positions are ignore_index).
"""
import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels,
                      T: float = 2.0, alpha: float = 0.5,
                      ignore_index: int = -100):
    mask = labels != ignore_index
    if int(mask.sum()) == 0:
        return student_logits.sum() * 0.0          # keeps graph, value 0
    s = student_logits[mask]
    t = teacher_logits[mask]
    y = labels[mask]
    kd = F.kl_div(F.log_softmax(s / T, dim=-1),
                  F.softmax(t / T, dim=-1),
                  reduction="batchmean") * (T * T)
    ce = F.cross_entropy(s, y)
    return alpha * kd + (1.0 - alpha) * ce
