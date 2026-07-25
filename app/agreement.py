"""Inter-judge agreement metrics (ADR-0002): weighted Cohen's kappa + Kendall's tau."""
import math

from scipy.stats import kendalltau


def cohen_kappa_weighted(a: list[int], b: list[int], max_score: int = 10) -> float:
    """Quadratic-weighted Cohen's kappa over integer scores in [0, max_score]."""
    k = max_score + 1
    n = len(a)
    if n == 0:
        return 0.0
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1.0
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * row[i] * col[j] / n for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1.0 - num / den


def kendall_tau(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    tau, _ = kendalltau(a, b)
    return 0.0 if (tau is None or math.isnan(tau)) else float(tau)
