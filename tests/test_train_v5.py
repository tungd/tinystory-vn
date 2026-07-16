import pytest

from scripts.train_v5 import resolve_max_steps


def test_v5_steps_cover_requested_epochs():
    assert resolve_max_steps(1000, 64, 10, 0) == 160
    assert resolve_max_steps(1000, 64, 10, 12) == 12


def test_v5_steps_reject_invalid_sizes():
    with pytest.raises(ValueError):
        resolve_max_steps(0, 64, 10, 0)
