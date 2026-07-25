import re
from dataclasses import dataclass

from app.guardrail.wordlist import BANNED_WORDS
from app.guardrail.wordlist_en import BANNED_WORDS_EN


@dataclass
class OutputDecision:
    ok: bool
    reason: str


def check_output(text: str) -> OutputDecision:
    if not text.strip():
        return OutputDecision(False, "The model returned empty content.")

    lowered = text.lower()
    tokens = set(re.findall(r"\w+", lowered, flags=re.UNICODE))
    en_tokens = set(re.findall(r"[a-z']+", lowered))
    if tokens & BANNED_WORDS or en_tokens & BANNED_WORDS_EN:
        return OutputDecision(False, "The generated story contains inappropriate words.")
    if any(" " in bad and bad in lowered for bad in BANNED_WORDS):
        return OutputDecision(False, "The generated story contains inappropriate words.")

    return OutputDecision(True, "")


def check_output_en(text: str) -> OutputDecision:
    return check_output(text)
