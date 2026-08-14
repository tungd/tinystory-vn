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

# Per-model prompt contracts. These are kept next to the shared English prompt
# builders so the global benchmark and Demo Studio cannot silently drift apart.
SLM_STORY_SEPARATOR = "<|story|>"
SMOLLM2_TF1_SYSTEM_MESSAGE = (
    "You are a world-class creative assistant that generates captivating and morally-driven "
    "fables based on structured inputs. Each fable must be: - Imaginative and coherent. "
    "- Appropriate for a wide audience, including young readers. - Structured around a "
    "classic fable format (character, setting, conflict, resolution, and moral). Age groups "
    "are defined as: A (≤3), B (4–7), C (8–11), D (12–15), E (≥16)."
)
TF1_FABLE_PROMPT = (
    "Create a fable based on the following elements. Weave them naturally into a story:\n"
    "- Main Character: {character}\n"
    "- Setting: {setting}\n"
    "- Challenge: {challenge}\n"
    "- Outcome: {outcome}\n"
    "- Teaching: {teaching}\n"
    "The fable should: be appropriate for age group B (4–7), use simple vocabulary, "
    "begin with vivid scene-setting, not use names (use trait + character), include simple "
    "dialogue, show (don't tell) growth, end with a clear connection to the moral. Keep it "
    "around 250 words."
)

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


def build_slm_prompt(
    character="", setting="", challenge="", outcome="", teaching="", length_hint=""
) -> str:
    """Build the SLM prompt plus the separator it saw during training."""

    return (
        build_fable_prompt(character, setting, challenge, outcome, teaching, length_hint)
        + "\n"
        + SLM_STORY_SEPARATOR
    )


def build_v16_prompt(character="", teaching="") -> str:
    """Build the two-field codec used by the V16 conditioned checkpoint."""

    return (
        f"<char> {character.strip()} </char>\n"
        f"<moral> {teaching.strip()} </moral>\n"
        "<story>\n"
    )


def build_smollm2_prompt(
    character="", setting="", challenge="", outcome="", teaching=""
) -> str:
    """Replay SmolLM2 LoRA C's TF1 system + five-slot training context."""

    prompt = TF1_FABLE_PROMPT.format(
        character=character.strip(),
        setting=setting.strip(),
        challenge=challenge.strip(),
        outcome=outcome.strip(),
        teaching=teaching.strip(),
    )
    return (
        f"{SMOLLM2_TF1_SYSTEM_MESSAGE.strip()}\n\n"
        f"{prompt}\n\n"
    )


def build_evaluation_brief(
    character="", setting="", challenge="", outcome="", teaching="", length_hint=""
) -> str:
    """Describe requested story elements without generation-format rules."""

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
    brief = "Evaluate the story against these requested narrative elements."
    if parts:
        brief += "\n" + "\n".join(parts)
    if length_hint:
        brief += "\n" + length_hint.strip()
    return brief
