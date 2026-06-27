import re
from dataclasses import dataclass

from app.guardrail.wordlist import BANNED_WORDS, OUT_OF_SCOPE_PATTERNS


@dataclass
class InputDecision:
    allowed: bool
    reason: str
    category: str


def _contains_banned(text: str) -> bool:
    tokens = set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
    if tokens & BANNED_WORDS:
        return True
    return any(bad in text.lower() for bad in BANNED_WORDS if " " in bad)


def check_input(topic: str, moral: str, age_range: str) -> InputDecision:
    combined = f"{topic} {moral} {age_range}".lower()

    if not topic.strip() or not moral.strip() or not age_range.strip():
        return InputDecision(False, "Vui lòng nhập đủ chủ đề, bài học và độ tuổi.", "empty")

    if _contains_banned(combined):
        return InputDecision(
            False,
            "Yêu cầu chứa từ ngữ không phù hợp. Mình chỉ tạo truyện ngụ ngôn trong sáng cho trẻ em.",
            "profanity",
        )

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, combined):
            return InputDecision(
                False,
                "Mình chỉ có thể tạo truyện ngụ ngôn cho trẻ em, không xử lý yêu cầu này.",
                "out_of_scope",
            )

    return InputDecision(True, "", "ok")
