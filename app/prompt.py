SYSTEM_PROMPT_GUARDED = (
    "You are a storyteller who writes short fables for young children. "
    "Only write wholesome, age-appropriate fables with a clear moral at the end. "
    "Do not include adult, violent, profane, or scary content. "
    "If the request is outside children's fables, refuse politely in one sentence."
)

SYSTEM_PROMPT_MINIMAL = "You are a helpful storyteller."


def build_instruction(topic: str, moral: str, age_range: str, length_hint: str = "") -> str:
    base = (
        f"Write a children's fable about: {topic.strip()}. "
        f"Moral lesson: {moral.strip()}. "
        f"Suitable age range: {age_range.strip()}."
    )
    if length_hint:
        base += " " + length_hint.strip()
    return base
