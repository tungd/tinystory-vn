"""Reference-free evaluation metrics for generated fables (pure Python).

Used by the Colab eval notebook and locally by tests. No ML deps.
"""

import math
import re
from collections import Counter


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]


def _syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev:
            count += 1
        prev = is_v
    if count == 0:
        count = 1
    if word.endswith("e") and count > 1:
        count -= 1
    return count


def flesch_reading_ease(text: str) -> float:
    words = _words(text)
    sentences = _sentences(text)
    if not words or not sentences:
        return 0.0
    syll = sum(_syllables(w) for w in words)
    return (
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syll / len(words))
    )


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int = 1) -> float:
    all_n = []
    for t in texts:
        toks = _words(t)
        all_n.extend(_ngrams(toks, n))
    if not all_n:
        return 0.0
    return len(set(all_n)) / len(all_n)


def self_bleu(texts: list[str], n: int = 4) -> float:
    """Average n-gram BLEU of each story against the rest (lower = more diverse)."""
    if len(texts) < 2:
        return 0.0
    scores = []
    for i, t in enumerate(texts):
        ref = _words(" ".join(texts[j] for j in range(len(texts)) if j != i))
        hyp = _words(t)
        if not hyp or not ref:
            continue
        precisions = []
        for k in range(1, n + 1):
            hyp_n = _ngrams(hyp, k)
            ref_n = Counter(_ngrams(ref, k))
            if not hyp_n:
                precisions.append(0.0)
                continue
            hit = sum(1 for g in hyp_n if ref_n.get(g, 0) > 0)
            precisions.append(hit / len(hyp_n))
        if all(p > 0 for p in precisions):
            scores.append(math.exp(sum(math.log(p) for p in precisions) / len(precisions)))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_metrics(texts: list[str]) -> dict:
    return {
        "n": len(texts),
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "self_bleu": round(self_bleu(texts), 4),
        "flesch_reading_ease": round(sum(flesch_reading_ease(t) for t in texts) / max(1, len(texts)), 2),
    }
