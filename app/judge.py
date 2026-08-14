import json
import re

from app import ollama_client

AXES = ["grammar", "creativity", "moral_clarity", "prompt_adherence"]

# Rubric 4 trục theo phương pháp TF1-EN-3M (Nadas et al. 2025, arXiv:2504.20605);
# không tự chế trục (ADR-0002). Mỗi trục chấm thang 1-10.
AXIS_RUBRIC = {
    "grammar": "Grammar & style: câu đúng ngữ pháp, mạch lạc, văn phong phù hợp truyện thiếu nhi.",
    "creativity": "Creativity: tình tiết mới mẻ, hình ảnh sinh động, không sáo mòn.",
    "moral_clarity": "Moral clarity: bài học đạo đức rõ ràng, được thể hiện qua cốt truyện.",
    "prompt_adherence": "Prompt adherence: bám đúng các yếu tố yêu cầu (nhân vật/bối cảnh/thử thách/kết cục/bài học).",
}


def build_google_judge_prompt(story: str, prompt: str) -> str:
    """Build the shared compact rubric used by the Google judges.

    Generation-format details are deliberately excluded from ``prompt`` by the
    callers. A literal ``Moral:`` footer is a model-format diagnostic, not a
    quality axis, so it must not affect the score.
    """

    return (
        "Judge this children's fable strictly on four 1-10 integer scales. "
        "Grammar covers grammatical correctness, coherence, and child-appropriate style. "
        "Creativity covers originality and vivid storytelling. Moral clarity covers whether "
        "the requested lesson is clearly earned by the plot. Prompt adherence covers the "
        "requested character, setting, challenge, outcome, and teaching as represented in "
        "the story. Do not reward or penalize a literal `Moral:` footer, its exact wording, "
        "or whether it appears; judge the lesson from the events. "
        "Return only one JSON object with exactly these integer keys: grammar, creativity, "
        "moral_clarity, prompt_adherence. Do not include reasons or markdown.\n\n"
        f"REQUEST:\n{prompt}\n\nSTORY:\n{story}\n\nJSON:"
    )


def build_google_judge_schema() -> dict:
    """Return the strict four-axis JSON schema shared by Google judge clients."""

    return {
        "type": "object",
        "properties": {
            axis: {"type": "integer", "minimum": 1, "maximum": 10}
            for axis in AXES
        },
        "required": AXES,
        "additionalProperties": False,
    }


def build_judge_prompt(story: str, prompt: str) -> str:
    return (
        "You are a strict judge of children's fables. Given the REQUEST and the STORY, "
        "rate the STORY from 1 to 10 on four axes: grammar, creativity, moral clarity, "
        "prompt adherence. For EACH axis return an object with an integer \"score\" (1-10) "
        "and a short \"reason\" (one sentence, citing specific evidence quoted from the STORY). "
        "Respond ONLY with a JSON object using keys "
        '"grammar","creativity","moral_clarity","prompt_adherence", e.g.\n'
        '{"grammar": {"score": 9, "reason": "Sentences are well-formed, e.g. \'...\'"}, '
        '"creativity": {"score": 8, "reason": "..."}, '
        '"moral_clarity": {"score": 10, "reason": "The moral is explicit: \'...\'"}, '
        '"prompt_adherence": {"score": 10, "reason": "Includes the requested elements: ..."}}\n\n'
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


def scores_from_google_response(response) -> tuple[dict, str]:
    """Read either Gemini's structured ``parsed`` value or its text fallback."""

    parsed = getattr(response, "parsed", None)
    raw = getattr(response, "text", "") or ""
    if parsed is not None and hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    if isinstance(parsed, dict):
        raw = json.dumps(parsed, ensure_ascii=False)
    return parse_scores(raw), raw


def validate_google_scores(scores: dict) -> dict:
    """Reject incomplete/out-of-range Google responses instead of scoring zero."""

    invalid = [
        axis
        for axis in AXES
        if not isinstance(scores.get(axis), int)
        or isinstance(scores.get(axis), bool)
        or not 1 <= scores[axis] <= 10
    ]
    if invalid:
        raise ValueError(f"invalid scores for axes: {', '.join(invalid)}")
    return scores


def evaluate(story: str, prompt: str, model: str, gen=None) -> dict:
    gen = gen or ollama_client.generate
    raw = gen(
        prompt=build_judge_prompt(story, prompt),
        system="You are a strict, fair evaluator. Output JSON only.",
        model=model,
    )
    return parse_scores(raw)
