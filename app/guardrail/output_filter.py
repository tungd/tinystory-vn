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
        return OutputDecision(False, "Mô hình trả về nội dung rỗng.")

    lowered = text.lower()
    tokens = set(re.findall(r"\w+", lowered, flags=re.UNICODE))
    if tokens & BANNED_WORDS or any(" " in bad and bad in lowered for bad in BANNED_WORDS):
        return OutputDecision(False, "Truyện sinh ra chứa từ ngữ không phù hợp.")

    return OutputDecision(True, "")


def check_output_en(text: str) -> OutputDecision:
    if not text.strip():
        return OutputDecision(False, "The model returned empty content.")
    toks = set(re.findall(r"[a-z']+", text.lower()))
    if toks & BANNED_WORDS_EN:
        return OutputDecision(False, "The generated story contains inappropriate words.")
    return OutputDecision(True, "")
