import json
import re

from app import ollama_client

AXES = ["grammar", "creativity", "moral_clarity", "prompt_adherence"]


def build_judge_prompt(story: str, prompt: str) -> str:
    return (
        "You are a strict judge of children's fables. Given the REQUEST and the STORY, "
        "rate the STORY from 1 to 10 on four axes: grammar, creativity, moral clarity, "
        "prompt adherence. Respond ONLY with a JSON object with keys "
        '"grammar","creativity","moral_clarity","prompt_adherence" (integers 1-10).\n\n'
        f"REQUEST:\n{prompt}\n\nSTORY:\n{story}\n\nJSON:"
    )


def parse_scores(raw: str) -> dict:
    m = re.search(r"\{[^{}]*\}", raw, re.S)
    data = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = {}
    out = {}
    for a in AXES:
        v = data.get(a, 0)
        try:
            out[a] = int(v)
        except (TypeError, ValueError):
            out[a] = 0
    out["overall"] = round(sum(out[a] for a in AXES) / len(AXES), 2)
    return out


def evaluate(story: str, prompt: str, model: str, gen=None) -> dict:
    gen = gen or ollama_client.generate
    raw = gen(
        prompt=build_judge_prompt(story, prompt),
        system="You are a strict, fair evaluator. Output JSON only.",
        model=model,
    )
    return parse_scores(raw)
