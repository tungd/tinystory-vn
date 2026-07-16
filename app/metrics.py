"""Reference-free diversity + readability metrics (ADR-0002)."""
from collections import Counter
from itertools import combinations

import textstat


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        grams = _ngrams(t.split(), n)
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def _bleu_n(cand: list[str], ref: list[str], n: int) -> float:
    if len(cand) < n:
        return 0.0
    cg, rg = Counter(_ngrams(cand, n)), Counter(_ngrams(ref, n))
    overlap = sum((cg & rg).values())
    return overlap / max(1, sum(cg.values()))


def self_bleu(texts: list[str], n: int = 4) -> float:
    toks = [t.split() for t in texts]
    pairs = list(combinations(range(len(toks)), 2))
    if not pairs:
        return 0.0
    return sum(_bleu_n(toks[i], toks[j], n) for i, j in pairs) / len(pairs)


def flesch_reading_ease(text: str) -> float:
    return float(textstat.flesch_reading_ease(text))
