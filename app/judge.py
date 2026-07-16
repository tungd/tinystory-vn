import json
import re

from app import ollama_client

AXES = ["grammar", "creativity", "moral_clarity", "prompt_adherence"]
PROMPT_VERSION = "v2-strict"
SYSTEM_INSTRUCTION = (
    "You are a skeptical, exacting evaluator of machine-generated children's "
    "fables. Judge only what is present in the story, apply the full 1-10 scale, "
    "and do not reward literal keyword inclusion as narrative success. Output JSON only."
)


def build_judge_prompt(story: str, prompt: str) -> str:
    return (
        "Evaluate the STORY against the REQUEST using these strict rules:\n\n"
        "SCORE ANCHORS\n"
        "- 9-10: exceptional and nearly flawless; reserve these scores.\n"
        "- 7-8: clearly good, with only minor defects.\n"
        "- 5-6: mixed quality with visible defects.\n"
        "- 3-4: poor, incoherent, or substantially mismatched.\n"
        "- 1-2: broken or unusable.\n\n"
        "AXES\n"
        "- grammar: Penalize malformed sentences, unclear pronouns, tense errors, "
        "formatting debris, duplicated morals, and awkward wording. A 9-10 requires "
        "essentially error-free prose.\n"
        "- creativity: Reward a distinctive but coherent plot. Penalize generic templates, "
        "repetition, unexplained objects or characters, and disconnected events.\n"
        "- moral_clarity: Judge whether the conflict, choices, and outcome causally demonstrate "
        "the requested moral. Merely appending the moral sentence is not enough; if the events "
        "teach a different lesson, score at most 4.\n"
        "- prompt_adherence: The requested character must be the active protagonist and the "
        "requested moral must shape the plot. Literal phrase inclusion alone scores at most 5.\n\n"
        "For EACH axis return an integer score from 1 to 10 and one concise reason citing "
        "specific evidence from the STORY. Respond ONLY with a JSON object using keys "
        '"grammar","creativity","moral_clarity","prompt_adherence", e.g.\n'
        '{"grammar":{"score":6,"reason":"..."},'
        '"creativity":{"score":5,"reason":"..."},'
        '"moral_clarity":{"score":3,"reason":"..."},'
        '"prompt_adherence":{"score":4,"reason":"..."}}\n\n'
        f"REQUEST:\n{prompt}\n\nSTORY:\n{story}\n\nJSON:"
    )


def _extract_json(raw: str) -> dict:
    """Extract the first balanced JSON object from raw text (tolerates nesting)."""
    start = raw.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _axis_from_json(data: dict, axis: str) -> tuple[int, str]:
    """Read (score, reason) for one axis from a parsed JSON dict."""
    v = data.get(axis)
    if isinstance(v, dict):
        reason = str(v.get("reason", "")).strip()
        raw_score = v.get("score", 0)
    else:
        reason = ""
        raw_score = v if v is not None else 0
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 0
    return score, reason


def _axis_from_regex(raw: str, axis: str) -> tuple[int, str]:
    """Tolerant per-axis extraction for when the model emits malformed JSON.

    Handles the common case where one axis object has a stray bracket that
    breaks whole-object ``json.loads`` but the individual fields are intact.
    """
    score = 0
    reason = ""
    # Nested: "axis": { ... "score": N ... "reason": "..." ... }
    m_score = re.search(rf'"{axis}"\s*:\s*\{{[^{{}}]*?"score"\s*:\s*(\d+)', raw, re.S)
    if m_score:
        score = int(m_score.group(1))
    else:
        # Flat legacy: "axis": N
        m_flat = re.search(rf'"{axis}"\s*:\s*(\d+)', raw)
        if m_flat:
            score = int(m_flat.group(1))
    m_reason = re.search(
        rf'"{axis}"\s*:\s*\{{[^{{}}]*?"reason"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.S
    )
    if m_reason:
        reason = m_reason.group(1).replace('\\"', '"').replace("\\'", "'").strip()
    return score, reason


def parse_scores(raw: str) -> dict:
    """Parse judge output into scores + per-axis rationale.

    Accepts either the nested shape ``{"grammar": {"score": 9, "reason": "..."}}``
    or the flat legacy shape ``{"grammar": 9}``. Robust to malformed JSON: falls
    back to tolerant per-axis regex extraction when whole-object parsing fails or
    yields a zero score. Missing axes default to 0. Returns flat integer scores
    per axis, ``overall`` (mean), and ``rationale`` (a dict mapping each axis to
    its reason string, omitting empty reasons).
    """
    data = _extract_json(raw)
    out: dict = {}
    rationale: dict = {}
    for a in AXES:
        score, reason = _axis_from_json(data, a)
        if score == 0 or not reason:
            r_score, r_reason = _axis_from_regex(raw, a)
            if score == 0:
                score = r_score
            if not reason:
                reason = r_reason
        out[a] = score
        if reason:
            rationale[a] = reason
    out["overall"] = round(sum(out[a] for a in AXES) / len(AXES), 2)
    out["rationale"] = rationale
    return out


def evaluate(story: str, prompt: str, model: str, gen=None) -> dict:
    gen = gen or ollama_client.generate
    raw = gen(
        prompt=build_judge_prompt(story, prompt),
        system=SYSTEM_INSTRUCTION,
        model=model,
        is_judge=True,
        num_predict=2000,
        temperature=0.1,
    )
    return parse_scores(raw)
