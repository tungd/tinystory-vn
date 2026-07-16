import math
from app.perplexity import perplexity_from_nll, aggregate_nll

def test_perplexity_uniform():
    # nll per token = ln(2) -> perplexity = 2
    assert abs(perplexity_from_nll(math.log(2)*10, 10) - 2.0) < 1e-6

def test_perplexity_zero_tokens_safe():
    assert perplexity_from_nll(0.0, 0) == float("inf")

def test_aggregate_nll_token_weighted():
    assert aggregate_nll([(2.0, 2), (3.0, 3)]) == 5.0
