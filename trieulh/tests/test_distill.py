import pytest
torch = pytest.importorskip("torch")
from trieulh.scripts.distill import distillation_loss


def test_all_ignored_returns_zero():
    s = torch.randn(3, 5); t = torch.randn(3, 5)
    y = torch.full((3,), -100)
    assert float(distillation_loss(s, t, y)) == 0.0


def test_matching_teacher_student_loss_is_small():
    # identical logits + labels = argmax -> KD term ~0, CE small
    logits = torch.tensor([[10.0, 0, 0], [0, 10.0, 0]])
    y = torch.tensor([0, 1])
    loss = float(distillation_loss(logits, logits.clone(), y, T=2.0, alpha=0.5))
    assert loss < 0.1


def test_alpha_blends_kd_and_ce():
    s = torch.randn(4, 6); t = torch.randn(4, 6); y = torch.randint(0, 6, (4,))
    only_ce = float(distillation_loss(s, t, y, alpha=0.0))
    blended = float(distillation_loss(s, t, y, alpha=0.5))
    assert only_ce >= 0 and blended >= 0 and blended != only_ce
