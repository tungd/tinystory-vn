import re
from dataclasses import dataclass

from app.guardrail.wordlist import BANNED_WORDS, OUT_OF_SCOPE_PATTERNS
from app.guardrail.wordlist_en import BANNED_WORDS_EN, OUT_OF_SCOPE_PATTERNS_EN


@dataclass
class InputDecision:
    allowed: bool
    reason: str
    category: str


def _contains_banned(text: str) -> bool:
    lowered = text.lower()
    tokens = set(re.findall(r"\w+", lowered, flags=re.UNICODE))
    en_tokens = set(re.findall(r"[a-z']+", lowered))
    if tokens & BANNED_WORDS or en_tokens & BANNED_WORDS_EN:
        return True
    return any(bad in lowered for bad in BANNED_WORDS if " " in bad)


def check_input(topic: str, moral: str, age_range: str) -> InputDecision:
    combined = f"{topic} {moral} {age_range}".lower()

    if not topic.strip() or not moral.strip() or not age_range.strip():
        return InputDecision(False, "Please enter a topic, moral, and age range.", "empty")

    if _contains_banned(combined):
        return InputDecision(
            False,
            "This request contains inappropriate words. I only write wholesome children's fables.",
            "profanity",
        )

    for pattern in OUT_OF_SCOPE_PATTERNS_EN + OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, combined):
            return InputDecision(
                False,
                "I can only write children's fables, not this request.",
                "out_of_scope",
            )

    return InputDecision(True, "", "ok")


def _contains_banned_en(text: str) -> bool:
    toks = set(re.findall(r"[a-z']+", text.lower()))
    return bool(toks & BANNED_WORDS_EN)


def check_input_en(character="", setting="", challenge="", outcome="", teaching="") -> InputDecision:
    combined = " ".join([character, setting, challenge, outcome, teaching]).lower()
    if _contains_banned_en(combined) or _contains_banned(combined):
        return InputDecision(False, "This request contains inappropriate words. I only write wholesome children's fables.", "profanity")
    for pat in OUT_OF_SCOPE_PATTERNS_EN:
        if re.search(pat, combined):
            return InputDecision(False, "I can only write children's fables, not this request.", "out_of_scope")
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, combined):
            return InputDecision(False, "I can only write children's fables, not this request.", "out_of_scope")
    return InputDecision(True, "", "ok")
