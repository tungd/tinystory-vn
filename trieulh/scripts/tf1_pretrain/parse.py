"""Extract the 5 narrative scaffold slots from a TF1-EN-3M prompt string."""
import re

_LABELS = {
    "character": "Main Character",
    "setting": "Setting",
    "challenge": "Challenge",
    "outcome": "Outcome",
    "teaching": "Teaching",
}


def parse_slots(prompt: str) -> dict[str, str]:
    """Return the 5 app slots parsed from a TF1 prompt. Missing slot -> ""."""
    out: dict[str, str] = {}
    for key, label in _LABELS.items():
        # Match "- <Label>: <value>" up to end-of-line.
        m = re.search(rf"-\s*{re.escape(label)}\s*:\s*(.+)", prompt)
        out[key] = m.group(1).strip() if m else ""
    return out
