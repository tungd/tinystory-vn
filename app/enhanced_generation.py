"""Validation and post-processing for enhanced English fable generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.prompt_en import SYSTEM_PROMPT_EN


MORAL_RE = re.compile(r"(?im)^\s*Moral\s*:\s*(.*)$")


@dataclass(frozen=True)
class StoryValidation:
    ok: bool
    fixable_reasons: list[str] = field(default_factory=list)
    severe_reasons: list[str] = field(default_factory=list)

    @property
    def needs_fix(self) -> bool:
        return bool(self.fixable_reasons)

    @property
    def needs_rewrite(self) -> bool:
        return bool(self.severe_reasons)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def _sentence_lengths(story: str) -> list[int]:
    body = MORAL_RE.sub("", story)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    return [len(_words(sentence)) for sentence in sentences]


def _paragraph_lengths(story: str) -> list[int]:
    paragraphs = [p.strip() for p in story.split("\n\n") if p.strip()]
    return [len(_words(paragraph)) for paragraph in paragraphs]


def normalize_moral_line(story: str, teaching: str) -> str:
    """Ensure the fable has exactly one final Moral line.

    This deliberately fixes format only. It does not rewrite the narrative.
    """
    cleaned = story.strip()
    teaching = teaching.strip()
    def punctuate(text: str) -> str:
        text = text.strip()
        return text if not text or text.endswith((".", "!", "?")) else f"{text}."

    fallback = (
        f"Moral: {punctuate(teaching)}"
        if teaching
        else "Moral: Be kind and learn from each day."
    )

    matches = list(MORAL_RE.finditer(cleaned))
    if not matches:
        return f"{cleaned.rstrip()}\n\n{fallback}".strip()

    last = matches[-1]
    moral_text = last.group(1).strip()
    final_moral = f"Moral: {punctuate(moral_text)}" if moral_text else fallback

    before = cleaned[: last.start()].rstrip()
    return f"{before}\n\n{final_moral}".strip()


def validate_story(story: str, prompt_row: dict) -> StoryValidation:
    fixable: list[str] = []
    severe: list[str] = []
    cleaned = story.strip()
    lower = cleaned.lower()
    words = _words(cleaned)

    moral_matches = list(MORAL_RE.finditer(cleaned))
    if not moral_matches:
        fixable.append("missing_moral")
    elif not moral_matches[-1].group(1).strip():
        fixable.append("empty_moral")
    elif cleaned[moral_matches[-1].end() :].strip():
        fixable.append("moral_not_final")

    if len(words) < 80:
        severe.append("too_short")
    if len(words) > 700:
        severe.append("too_long")
    if re.search(r"^\s*[-*]\s+", cleaned, flags=re.M):
        severe.append("contains_bullets")
    sentence_lengths = _sentence_lengths(cleaned)
    if sentence_lengths and max(sentence_lengths) > 65:
        severe.append("run_on_sentence")
    paragraph_lengths = _paragraph_lengths(cleaned)
    if paragraph_lengths and max(paragraph_lengths) > 180:
        severe.append("paragraph_too_long")

    for key in ("character", "setting"):
        value = str(prompt_row.get(key, ""))
        keyword = value.split()[-1].lower() if value.split() else ""
        if keyword and keyword not in lower:
            severe.append(f"{key}_keyword_missing")

    return StoryValidation(ok=not fixable and not severe, fixable_reasons=fixable, severe_reasons=severe)


def build_rewrite_prompt(story: str, prompt_row: dict) -> str:
    teaching = prompt_row.get("teaching", "")
    return (
        "Rewrite the draft into a polished children's fable.\n"
        "Keep the same requested elements and do not add unrelated events.\n"
        "Use clear, simple English with 2-4 short paragraphs.\n"
        "Remove bullet lists, run-on sentences, repeated ideas, and confusing logic.\n"
        "End with exactly one final Moral line.\n\n"
        "Requested elements:\n"
        f"- Character: {prompt_row.get('character', '')}\n"
        f"- Setting: {prompt_row.get('setting', '')}\n"
        f"- Challenge: {prompt_row.get('challenge', '')}\n"
        f"- Outcome: {prompt_row.get('outcome', '')}\n"
        f"- Teaching/Moral: {teaching}\n\n"
        "Draft story:\n"
        f"{story.strip()}\n\n"
        f"Final line must be exactly: Moral: {teaching}"
    )


def enhance_story(
    story: str,
    prompt_row: dict,
    *,
    rewrite_model: str | None = None,
    generate_meta_fn=None,
    seed: int | None = None,
) -> dict:
    """Validate, post-process, and optionally rewrite a story.

    Returns a dict with the enhanced story and metadata describing actions.
    """
    initial_validation = validate_story(story, prompt_row)
    actions: list[str] = []
    latency_ms = 0
    rewrite_meta: dict = {}
    enhanced = story

    if initial_validation.needs_rewrite and rewrite_model and generate_meta_fn:
        try:
            result = generate_meta_fn(
                prompt=build_rewrite_prompt(story, prompt_row),
                system=SYSTEM_PROMPT_EN,
                model=rewrite_model,
                num_predict=500,
                seed=seed,
                temperature=0.4,
                top_p=0.9,
                repeat_penalty=1.15,
            )
            enhanced = result["text"]
            latency_ms += result.get("latency_ms", 0)
            rewrite_meta = {
                "model": rewrite_model,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "latency_ms": result.get("latency_ms", 0),
            }
            actions.append("rewrite")
        except Exception as exc:  # pragma: no cover - exercised by live Ollama failures
            rewrite_meta = {"model": rewrite_model, "error": str(exc)}
            actions.append("rewrite_failed")

    fixed = normalize_moral_line(enhanced, str(prompt_row.get("teaching", "")))
    if fixed != enhanced.strip():
        actions.append("moral_postprocess")
    enhanced = fixed
    final_validation = validate_story(enhanced, prompt_row)

    return {
        "story": enhanced,
        "initial_validation": {
            "ok": initial_validation.ok,
            "fixable_reasons": initial_validation.fixable_reasons,
            "severe_reasons": initial_validation.severe_reasons,
        },
        "final_validation": {
            "ok": final_validation.ok,
            "fixable_reasons": final_validation.fixable_reasons,
            "severe_reasons": final_validation.severe_reasons,
        },
        "actions": actions,
        "rewrite_meta": rewrite_meta,
        "extra_latency_ms": latency_ms,
    }
