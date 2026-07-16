"""Conditional training-text formatting for SLM pretraining."""
import random

from app.prompt_en import build_fable_prompt, LENGTH_HINT_EN

SEP = "<|story|>"
END = "<|end|>"
PAD = "<|pad|>"
SLOT_KEYS = ("character", "setting", "challenge", "outcome", "teaching")


def length_bucket(word_count: int) -> str:
    if word_count < 200:
        return "short"
    if word_count < 450:
        return "medium"
    return "long"


def apply_dropout(slots: dict, rng: random.Random, p: float = 0.3,
                  p_all: float = 0.05,
                  p_overrides: dict | None = None) -> dict:
    """Return a copy of slots with some values blanked (keys preserved).

    With probability p_all, blank every slot (free-generation example).
    Otherwise blank each slot independently with probability p, or with
    p_overrides[slot] when given (e.g. lower dropout for teaching/outcome so
    the model learns to FOLLOW the requested moral instead of inventing one).
    """
    if rng.random() < p_all:
        return {k: "" for k in slots}
    ov = p_overrides or {}
    return {k: ("" if rng.random() < ov.get(k, p) else v)
            for k, v in slots.items()}


def build_training_text(slots: dict, fable: str, length: str) -> tuple[str, int]:
    """Return (text, cond_len_chars). text = COND + SEP + fable + END.

    Conditioning reuses the app's build_fable_prompt so training format
    matches inference. cond_len_chars is the index where SEP begins (for
    loss masking the conditioning region).
    """
    cond = build_fable_prompt(
        slots.get("character", ""), slots.get("setting", ""),
        slots.get("challenge", ""), slots.get("outcome", ""),
        slots.get("teaching", ""), LENGTH_HINT_EN[length],
    )
    prefix = cond + "\n"
    text = prefix + SEP + fable.strip() + END
    return text, len(prefix)
