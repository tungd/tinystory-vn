"""Held-out perplexity aggregation (mockable; model forward done by caller)."""
import math


def aggregate_nll(per_seq: list[tuple[float, int]]) -> float:
    return sum(nll for nll, _ in per_seq)


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
