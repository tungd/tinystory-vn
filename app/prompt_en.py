SYSTEM_PROMPT_EN = (
    "You are a storyteller who writes short fables for young children (ages 4-7). "
    "Write a single, coherent fable in simple English with a clear moral at the end. "
    "Keep it wholesome and age-appropriate. If the request is not about writing a "
    "children's fable, politely refuse in one sentence."
)
SYSTEM_PROMPT_MINIMAL_EN = "You are a helpful storyteller."

LENGTH_NUM_PREDICT = {"short": 150, "medium": 350, "long": 600}
LENGTH_HINT_EN = {
    "short": "Keep it very short (about 80-120 words).",
    "medium": "Write a medium-length fable (about 180-250 words).",
    "long": "Write a longer fable (about 350-450 words).",
}

_LABELS = [
    ("character", "Main character"),
    ("setting", "Setting"),
    ("challenge", "Challenge"),
    ("outcome", "Outcome"),
    ("teaching", "Teaching/Moral"),
]


SEED_PREFIX = (
    "<char> {character} </char>\n"
    "<moral> {moral} </moral>\n"
    "<story>\n"
)


def build_seed_prompt(character="", moral="", length_hint="") -> str:
    """Keyword-guided prefix for the from-scratch 200M fable model.

    Seed elements only: main character + moral lesson. The model was trained to
    continue from this prefix and emit the fable body. Matches the format produced
    by `scripts.prepare_tf1.format_sample`.
    """
    base = SEED_PREFIX.format(character=character.strip(), moral=moral.strip())
    if length_hint:
        base += length_hint.strip() + "\n"
    return base


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
