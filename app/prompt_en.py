SYSTEM_PROMPT_EN = (
    "You are a storyteller who writes short fables for young children (ages 4-7). "
    "Write a single, coherent fable in simple English with a clear moral at the end. "
    "Keep it wholesome and age-appropriate. If the request is not about writing a "
    "children's fable, politely refuse in one sentence."
)
SYSTEM_PROMPT_MINIMAL_EN = "You are a helpful storyteller."

LENGTH_NUM_PREDICT = {"short": 300, "medium": 600, "long": 1000}
LENGTH_HINT_EN = {
    "short": "Keep it very short (about 120-180 words).",
    "medium": "Write a medium-length fable (about 250-350 words).",
    "long": "Write a longer fable (about 450-600 words).",
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
    base = "Write a children's fable."
    if parts:
        base += " Use these narrative elements:\n" + "\n".join(parts)
    if length_hint:
        base += "\n" + length_hint.strip()
    return base
