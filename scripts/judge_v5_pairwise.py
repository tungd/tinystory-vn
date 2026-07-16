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
        '"winner":"a|b|tie","confidence":1,"reason":"specific comparative evidence"}\n\n'
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    selected = select_sources(data["generations"], args.controls, args.seed)
    pairs = {}
    for row in data["generations"]:
        if row["source"] in selected:
            pairs.setdefault(row["source"], {})[row["model"]] = row
    judgments = []
    for index, source in enumerate(selected, 1):
        pair = pairs[source]
        model_a, model_b = blind_order(source, args.seed)
        request = f"Main character: {pair[model_a]['character']}\nTeaching: {pair[model_a]['moral']}"
        started = time.perf_counter()
        raw = google_judge_client.generate(
            prompt=build_prompt(request, pair[model_a]["story"], pair[model_b]["story"]),
            system=SYSTEM_INSTRUCTION,
            model=args.model,
            num_predict=2000,
            temperature=0.0,
        )
        parsed = parse_pairwise(raw)
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
        print(f"judged {index}/{len(selected)}", flush=True)
    result = {
        "kind": "v5-blind-pairwise-judge",
        "source": args.input,
        "selection": {"controls": len(selected), "seed": args.seed},
        "judge_settings": {
            "backend": "google-genai",
            "model": args.model,
            "thinking_level": google_judge_client.JUDGE_THINKING_LEVEL,
            "prompt_version": "v3-blind-pairwise-strict",
        },
        "summary": summarize(judgments),
        "judgments": judgments,
    }
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
