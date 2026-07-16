from app.agreement import cohen_kappa_weighted, kendall_tau

def test_kappa_perfect_agreement():
    assert cohen_kappa_weighted([1,5,9], [1,5,9]) == 1.0

def test_kappa_close_scores_positive():
    assert cohen_kappa_weighted([8,9,7,10], [7,9,8,10]) > 0.0

def test_kendall_tau_monotonic():
    assert kendall_tau([1,2,3,4], [1,2,3,4]) == 1.0

def test_kendall_tau_reversed():
    assert kendall_tau([1,2,3,4], [4,3,2,1]) == -1.0
