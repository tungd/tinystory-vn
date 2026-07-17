"""Blind, high-thinking paired judge for v3-full versus v5."""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import google_judge_client, judge


SYSTEM_INSTRUCTION = (
    "You are a skeptical children's-fiction editor conducting a blind paired review. "
    "Judge only the supplied stories and request. Do not infer model identity. Output JSON only."
)

SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["grammar", "creativity", "moral_clarity", "prompt_adherence"],
    "properties": {
        axis: {"type": "integer", "minimum": 1, "maximum": 10}
        for axis in judge.AXES
    },
}
PAIRWISE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a", "b", "winner", "confidence", "reason"],
    "properties": {
        "a": SCORE_SCHEMA,
        "b": SCORE_SCHEMA,
        "winner": {"type": "string", "enum": ["a", "b", "tie"]},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {
            "type": "string",
            "enum": [
                "grammar",
                "creativity",
                "moral_clarity",
                "prompt_adherence",
                "mixed",
            ],
        },
    },
}


def select_sources(rows: list[dict], count: int, seed: int) -> list[str]:
    sources = [row["source"] for row in rows if row["model"] == "v3-full"]
    return random.Random(seed).sample(sources, min(count, len(sources)))


def blind_order(source: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    return ("v3-full", "v5") if digest[0] % 2 == 0 else ("v5", "v3-full")


def build_prompt(request: str, story_a: str, story_b: str) -> str:
    return (
        "Compare STORY A and STORY B against the same REQUEST. The order is randomized.\n\n"
        "Score each story on grammar, creativity, moral_clarity, and prompt_adherence "
        "from 1-10 using the strict anchors: 9-10 exceptional, 7-8 good, 5-6 mixed, "
        "3-4 poor, 1-2 broken. Literal character or moral inclusion is not narrative "
        "success. The moral must emerge causally from conflict, choice, and outcome.\n\n"
        "Return JSON only:\n"
        '{"a":{"grammar":1,"creativity":1,"moral_clarity":1,"prompt_adherence":1},'
        '"b":{"grammar":1,"creativity":1,"moral_clarity":1,"prompt_adherence":1},'
        '"winner":"a|b|tie","confidence":1,'
        '"reason":"grammar|creativity|moral_clarity|prompt_adherence|mixed"}\n\n'
        "Reason is only the dominant deciding axis, never prose.\n\n"
        f"REQUEST:\n{request}\n\nSTORY A:\n{story_a}\n\nSTORY B:\n{story_b}\n\nJSON:"
    )


def parse_pairwise(raw: str) -> dict:
    data = judge._extract_json(raw)
    scores = {}
    for side in ("a", "b"):
        values = data.get(side, {})
        scores[side] = {}
        for axis in judge.AXES:
            try:
                score = int(values.get(axis, 0))
            except (TypeError, ValueError):
                score = 0
            if not 1 <= score <= 10:
                raise ValueError(f"Invalid pairwise {side}.{axis}: {score}")
            scores[side][axis] = score
        scores[side]["overall"] = round(
            sum(scores[side][axis] for axis in judge.AXES) / len(judge.AXES), 2
        )
    winner = str(data.get("winner", "")).casefold()
    if winner not in {"a", "b", "tie"}:
        raise ValueError(f"Invalid pairwise winner: {winner}")
    confidence = int(data.get("confidence", 0))
    if not 1 <= confidence <= 5:
        raise ValueError(f"Invalid pairwise confidence: {confidence}")
    return {
        "scores": scores,
        "winner": winner,
        "confidence": confidence,
        "reason": " ".join(str(data.get("reason", "")).split()),
    }


def summarize(judgments: list[dict]) -> dict:
    wins = Counter(row["winner_model"] for row in judgments)
    means = {}
    for model in ("v3-full", "v5"):
        means[model] = {
            axis: round(sum(row["scores"][model][axis] for row in judgments) / len(judgments), 2)
            for axis in judge.AXES + ["overall"]
        }
    return {"wins": dict(wins), "mean_scores": means}


def build_result(args, selected: list[str], judgments: list[dict]) -> dict:
    return {
        "kind": "v5-blind-pairwise-judge",
        "source": args.input,
        "selection": {"controls": len(selected), "seed": args.seed},
        "judge_settings": {
            "backend": "google-genai",
            "model": args.model,
            "thinking_level": google_judge_client.JUDGE_THINKING_LEVEL,
            "max_output_tokens": args.max_output_tokens,
            "prompt_version": "v3-blind-pairwise-strict-schema",
        },
        "complete": len(judgments) == len(selected),
        "summary": summarize(judgments) if judgments else {},
        "judgments": judgments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    selected = select_sources(data["generations"], args.controls, args.seed)
    pairs = {}
    for row in data["generations"]:
        if row["source"] in selected:
            pairs.setdefault(row["source"], {})[row["model"]] = row
    output = Path(args.out)
    judgments = []
    if output.exists():
        existing = json.loads(output.read_text())
        expected = {"controls": len(selected), "seed": args.seed}
        if existing.get("source") != args.input or existing.get("selection") != expected:
            raise ValueError("Existing pairwise output does not match this run")
        judgments = existing.get("judgments", [])
    completed = {row["source"] for row in judgments}
    for index, source in enumerate(selected, 1):
        if source in completed:
            print(f"judged {index}/{len(selected)} (resumed)", flush=True)
            continue
        pair = pairs[source]
        model_a, model_b = blind_order(source, args.seed)
        request = f"Main character: {pair[model_a]['character']}\nTeaching: {pair[model_a]['moral']}"
        started = time.perf_counter()
        for attempt in range(1, args.retries + 1):
            try:
                raw = google_judge_client.generate(
                    prompt=build_prompt(
                        request, pair[model_a]["story"], pair[model_b]["story"]
                    ),
                    system=SYSTEM_INSTRUCTION,
                    model=args.model,
                    num_predict=args.max_output_tokens,
                    temperature=0.0,
                    response_schema=PAIRWISE_RESPONSE_SCHEMA,
                )
                parsed = parse_pairwise(raw)
                break
            except Exception as error:
                if attempt == args.retries:
                    raise
                print(
                    f"retry {attempt}/{args.retries} for {source}: {type(error).__name__}",
                    flush=True,
                )
                time.sleep(2**attempt)
        winner_model = "tie" if parsed["winner"] == "tie" else {
            "a": model_a, "b": model_b
        }[parsed["winner"]]
        judgments.append({
            "source": source,
            "character": pair[model_a]["character"],
            "moral": pair[model_a]["moral"],
            "blind_order": {"a": model_a, "b": model_b},
            "scores": {model_a: parsed["scores"]["a"], model_b: parsed["scores"]["b"]},
            "winner_model": winner_model,
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "latency_ms": round((time.perf_counter() - started) * 1000),
        })
        output.write_text(
            json.dumps(build_result(args, selected, judgments), indent=2, ensure_ascii=False)
            + "\n"
        )
        print(f"judged {index}/{len(selected)}", flush=True)
    result = build_result(args, selected, judgments)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
