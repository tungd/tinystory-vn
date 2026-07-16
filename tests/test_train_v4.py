from scripts.train_v4 import resolve_max_steps


def test_v4_steps_cover_one_epoch():
    assert resolve_max_steps(250_000, 64, 0) == 3907
    assert resolve_max_steps(20_000, 64, 0) == 313


def test_explicit_v4_steps_override_auto_value():
    assert resolve_max_steps(250_000, 64, 100) == 100
