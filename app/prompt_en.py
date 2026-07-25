SYSTEM_PROMPT_EN = (
    "You are a storyteller who writes short fables for young children (ages 4-7). "
    "Write a single, coherent fable in simple English with a clear beginning, problem, "
    "resolution, and a clear moral at the end. "
    "The story is incomplete unless the final line starts exactly with 'Moral:'. "
    "Keep it wholesome and age-appropriate. Do not include adult, violent, profane, "
    "or scary content. If the request is not about writing a children's fable, "
    "politely refuse in one sentence."
)

SYSTEM_PROMPT_MINIMAL_EN = "You are a helpful storyteller who writes in simple English."

# num_predict sát headroom của context 512 (prompt ~50-110 token -> story ~400-460
# token). Bỏ 600/1000 cũ vì vượt seq_len train (model không dùng nổi, chỉ gây cụt).
LENGTH_NUM_PREDICT = {"short": 400, "medium": 440, "long": 460}
# Mốc từ kéo về mức model KẾT ĐƯỢC trong context (~300-340 từ tối đa). Hint "long
# 450-600" cũ vượt khả năng -> chính long dễ cụt nhất. Giữ gradient tương đối.
LENGTH_HINT_EN = {
    "short": "Keep it short (about 120-160 words).",
    "medium": "Write a medium-length fable (about 200-260 words).",
    "long": "Write a fuller fable (about 280-340 words).",
}

_LABELS = [
    ("character", "Main character"),
    ("setting", "Setting"),
    ("challenge", "Challenge"),
    ("outcome", "Outcome"),
    ("teaching", "Teaching/Moral"),
]


def build_fable_prompt(
    character="", setting="", challenge="", outcome="", teaching="", length_hint=""
) -> str:
    vals = {
        "character": character,
        "setting": setting,
        "challenge": challenge,
        "outcome": outcome,
        "teaching": teaching,
    }
    parts = [
        f"- {label}: {vals[key].strip()}"
        for key, label in _LABELS
        if vals[key].strip()
    ]
    base = (
        "Write one coherent children's fable. Keep the events connected and easy to follow. "
        "Do not stop until you write a final line that starts exactly with 'Moral:'."
    )
    if parts:
        base += " Use these narrative elements:\n" + "\n".join(parts)
    if length_hint:
        base += "\n" + length_hint.strip()
    return base
