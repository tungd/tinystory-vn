#!/usr/bin/env python3
"""No-retraining ablations for condition availability, causality, and repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import global_benchmark as gb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ablation_judge"
GEN_DIR = RESULTS / "generations"
LOG_DIR = RESULTS / "logs"
CF_PROMPTS = RESULTS / "counterfactual_prompts_v1.jsonl"
PROTOCOL = RESULTS / "protocol.json"
INDIVIDUAL_MAP = RESULTS / "individual_map.private.json"
INDIVIDUAL_SCORES = RESULTS / "individual_scores.blinded.jsonl"
PAIR_MAP = RESULTS / "pair_map.private.json"
PAIR_SCORES = RESULTS / "pair_scores.blinded.jsonl"
SUMMARY = RESULTS / "summary.json"
SUMMARY_MD = RESULTS / "summary.md"
RUN_MANIFEST = RESULTS / "run_manifest.json"

JUDGE_SEED = 20260727
BOOTSTRAP_ROUNDS = 10_000
INDIVIDUAL_FIELDS = [
    "character_covered",
    "setting_covered",
    "challenge_covered",
    "outcome_covered",
    "teaching_covered",
    "moral_footer_present",
    "trait_drives_choice",
    "choice_causes_outcome",
    "internal_causal_consistency",
    "requested_causal_consistency",
]


def counterfactual_pairs() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": "CF01",
            "intervention": "trait",
            "variant_a": {
                "character": "a cautious young rabbit who checks every foothold",
                "setting": "a narrow rope bridge above a swollen creek",
                "challenge": "medicine must cross before sunset after several bridge planks come loose",
                "outcome": "the rabbit reaches a sick hedgehog safely with the medicine",
                "teaching": "Courage works best when guided by care.",
            },
            "variant_b": {
                "character": "a reckless young rabbit who leaps before looking",
                "setting": "a narrow rope bridge above a swollen creek",
                "challenge": "medicine must cross before sunset after several bridge planks come loose",
                "outcome": "the rabbit reaches a sick hedgehog safely with the medicine",
                "teaching": "Courage works best when guided by care.",
            },
        },
        {
            "pair_id": "CF02",
            "intervention": "trait",
            "variant_a": {
                "character": "a generous squirrel who gladly shares her winter pantry",
                "setting": "an oak forest buried by an early snowstorm",
                "challenge": "several neighboring animals have no food left",
                "outcome": "the forest community shares enough food to survive until the thaw",
                "teaching": "What we share in hard times returns as shared strength.",
            },
            "variant_b": {
                "character": "a selfish squirrel who guards every acorn in her winter pantry",
                "setting": "an oak forest buried by an early snowstorm",
                "challenge": "several neighboring animals have no food left",
                "outcome": "the forest community shares enough food to survive until the thaw",
                "teaching": "What we share in hard times returns as shared strength.",
            },
        },
        {
            "pair_id": "CF03",
            "intervention": "trait",
            "variant_a": {
                "character": "a patient red panda baker",
                "setting": "a mountain village preparing for the first snow",
                "challenge": "the old oven heats unevenly and the festival bread will not rise",
                "outcome": "the final loaf feeds the village at the evening celebration",
                "teaching": "Patience gives good work time to grow.",
            },
            "variant_b": {
                "character": "an impatient red panda baker",
                "setting": "a mountain village preparing for the first snow",
                "challenge": "the old oven heats unevenly and the festival bread will not rise",
                "outcome": "the final loaf feeds the village at the evening celebration",
                "teaching": "Patience gives good work time to grow.",
            },
        },
        {
            "pair_id": "CF04",
            "intervention": "trait",
            "variant_a": {
                "character": "an honest young fox messenger",
                "setting": "a windy road between two hill villages",
                "challenge": "a sealed warning letter falls into a muddy ditch",
                "outcome": "the council receives the warning in time to reinforce the dam",
                "teaching": "Honesty protects trust when mistakes happen.",
            },
            "variant_b": {
                "character": "a deceptive young fox messenger who hides mistakes",
                "setting": "a windy road between two hill villages",
                "challenge": "a sealed warning letter falls into a muddy ditch",
                "outcome": "the council receives the warning in time to reinforce the dam",
                "teaching": "Honesty protects trust when mistakes happen.",
            },
        },
        {
            "pair_id": "CF05",
            "intervention": "trait",
            "variant_a": {
                "character": "a humble young crane competing in her first flying contest",
                "setting": "a lakeside meadow under gathering storm clouds",
                "challenge": "a rival bird injures a wing while everyone races home",
                "outcome": "the crane helps the rival land safely before the storm",
                "teaching": "True strength is shown by whom we choose to help.",
            },
            "variant_b": {
                "character": "a boastful young crane determined to win at any cost",
                "setting": "a lakeside meadow under gathering storm clouds",
                "challenge": "a rival bird injures a wing while everyone races home",
                "outcome": "the crane helps the rival land safely before the storm",
                "teaching": "True strength is shown by whom we choose to help.",
            },
        },
        {
            "pair_id": "CF06",
            "intervention": "outcome",
            "variant_a": {
                "character": "a resourceful young otter who collects smooth stones",
                "setting": "a misty river bend beside a willow tree",
                "challenge": "a sudden current carries away the markers for a safe path",
                "outcome": "the otter and river animals build a new path from interlocked branches",
                "teaching": "Shared ideas can turn a loss into a safer solution.",
            },
            "variant_b": {
                "character": "a resourceful young otter who collects smooth stones",
                "setting": "a misty river bend beside a willow tree",
                "challenge": "a sudden current carries away the markers for a safe path",
                "outcome": "the otter and river animals mark a new route with rope and woven reeds",
                "teaching": "Shared ideas can turn a loss into a safer solution.",
            },
        },
        {
            "pair_id": "CF07",
            "intervention": "outcome",
            "variant_a": {
                "character": "a shy firefly with a very small glow",
                "setting": "a moonless garden during a summer festival",
                "challenge": "the lanterns go out and the smallest insects cannot find their families",
                "outcome": "the firefly gathers friends so their tiny lights become a bright guide",
                "teaching": "Small gifts become powerful when we share them.",
            },
            "variant_b": {
                "character": "a shy firefly with a very small glow",
                "setting": "a moonless garden during a summer festival",
                "challenge": "the lanterns go out and the smallest insects cannot find their families",
                "outcome": "the firefly arranges dew-covered leaves to reflect every tiny light",
                "teaching": "Small gifts become powerful when we share them.",
            },
        },
        {
            "pair_id": "CF08",
            "intervention": "outcome",
            "variant_a": {
                "character": "a clever field mouse who dislikes getting dirty",
                "setting": "a flooded vegetable garden",
                "challenge": "the seedlings will drown before morning",
                "outcome": "the mouse joins the animals digging a muddy drainage channel",
                "teaching": "Useful work matters more than staying perfectly clean.",
            },
            "variant_b": {
                "character": "a clever field mouse who dislikes getting dirty",
                "setting": "a flooded vegetable garden",
                "challenge": "the seedlings will drown before morning",
                "outcome": "the mouse joins the animals moving every seedling onto raised soil mounds",
                "teaching": "Useful work matters more than staying perfectly clean.",
            },
        },
        {
            "pair_id": "CF09",
            "intervention": "outcome",
            "variant_a": {
                "character": "a careful young turtle carrying medicine",
                "setting": "a forest stream after a fallen bridge",
                "challenge": "the medicine must reach the far bank before night",
                "outcome": "the turtle organizes flat stones into a safe chain of stepping places",
                "teaching": "Careful planning can make a difficult crossing possible.",
            },
            "variant_b": {
                "character": "a careful young turtle carrying medicine",
                "setting": "a forest stream after a fallen bridge",
                "challenge": "the medicine must reach the far bank before night",
                "outcome": "the turtle builds a broad leaf raft and ferries the medicine across",
                "teaching": "Careful planning can make a difficult crossing possible.",
            },
        },
        {
            "pair_id": "CF10",
            "intervention": "outcome",
            "variant_a": {
                "character": "a practical beaver who listens to younger animals",
                "setting": "a woodland pond during a long drought",
                "challenge": "the remaining water will not last until the rains return",
                "outcome": "the animals reopen an old shaded canal from a spring",
                "teaching": "A community survives when every useful idea is heard.",
            },
            "variant_b": {
                "character": "a practical beaver who listens to younger animals",
                "setting": "a woodland pond during a long drought",
                "challenge": "the remaining water will not last until the rains return",
                "outcome": "the animals build covered cisterns to collect dew and brief showers",
                "teaching": "A community survives when every useful idea is heard.",
            },
        },
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def two_slot_prompt(row: dict[str, Any]) -> str:
    return (
        "Write one coherent children's fable. Keep the events connected and easy to follow. "
        "Do not stop until you write a final line that starts exactly with 'Moral:'. "
        "Use these narrative elements:\n"
        f"- Main character: {row['character']}\n"
        f"- Teaching/Moral: {row['teaching']}\n"
        "Write a medium-length fable (about 200-260 words)."
    )


def prepare() -> None:
    rows = []
    for pair in counterfactual_pairs():
        for variant in ("a", "b"):
            row = {
                "prompt_id": f"{pair['pair_id']}{variant.upper()}",
                "pair_id": pair["pair_id"],
                "variant": variant,
                "intervention": pair["intervention"],
                **pair[f"variant_{variant}"],
            }
            rows.append(row)
    CF_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    CF_PROMPTS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(
        PROTOCOL,
        {
            "version": "v1",
            "purpose": [
                "condition availability: five slots versus character+moral",
                "compositional adherence: exact semantic coverage per slot",
                "causal use: trait-to-choice and choice-to-outcome",
                "counterfactual sensitivity: matched trait/outcome interventions",
                "E4 repair contribution: raw versus final output",
            ],
            "models": ["e2", "e5"],
            "two_slot_prompts": 25,
            "counterfactual_pairs": 10,
            "counterfactual_prompts": 20,
            "counterfactual_split": {"trait": 5, "outcome": 5},
            "generation": {
                **gb.GENERATION,
                "two_slot_seed": "5410 + original prompt index",
                "counterfactual_seed": "16000 + pair index; shared by A/B and E1/E5",
            },
            "judge_seed": JUDGE_SEED,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        },
    )
    print(f"prepared {len(rows)} counterfactual prompts")


def generation_row(
    candidate: str,
    suite: str,
    prompt: dict[str, Any],
    model_prompt: str,
    seed: int,
    generated: dict[str, Any],
) -> dict[str, Any]:
    story = gb.clean_story(generated["text"])
    return {
        "candidate_id": candidate,
        "candidate_name": gb.CANDIDATES[candidate]["name"],
        "suite": suite,
        "prompt_id": prompt["prompt_id"],
        "pair_id": prompt.get("pair_id"),
        "variant": prompt.get("variant"),
        "intervention": prompt.get("intervention"),
        "seed": seed,
        "prompt": prompt,
        "formatted_prompt": gb.prompt_text(prompt),
        "model_prompt": model_prompt,
        "story": story,
        "word_count": gb.word_count(story),
        **generated,
    }


def generate(candidate: str, suite: str) -> None:
    if candidate not in {"e2", "e5"}:
        raise ValueError(candidate)
    if suite == "two_slot":
        prompts = gb.read_jsonl(gb.PROMPTS_PATH)
    else:
        prompts = gb.read_jsonl(CF_PROMPTS)
    output = GEN_DIR / f"{candidate}.{suite}.jsonl"
    completed = {row["prompt_id"] for row in gb.read_jsonl(output)}
    pending = [row for row in prompts if row["prompt_id"] not in completed]
    print(f"{candidate}/{suite}: {len(completed)} complete, {len(pending)} pending")
    if not pending:
        return
    port = {"e2": 18182, "e5": 18185}[candidate]
    model = Path(gb.CANDIDATES[candidate]["path"])
    with gb.llama_server(model, port, LOG_DIR / f"{candidate}.{suite}.log") as base_url:
        for row in pending:
            if suite == "two_slot":
                index = int(row["prompt_id"][-2:]) - 1
                seed = 5410 + index
                text = two_slot_prompt(row)
            else:
                index = int(row["pair_id"][-2:]) - 1
                seed = 16000 + index
                text = gb.prompt_text(row)
            if candidate == "e2":
                generated = gb.llama_completion(base_url, text, seed)
            else:
                generated = gb.llama_chat(base_url, text, seed)
            result = generation_row(candidate, suite, row, text, seed, generated)
            gb.append_jsonl(output, result)
            print(
                f"{candidate}/{suite} {row['prompt_id']}: "
                f"{result['word_count']} words, {result['latency_ms']} ms",
                flush=True,
            )


def individual_items() -> list[dict[str, Any]]:
    prompts = {row["prompt_id"]: row for row in gb.read_jsonl(gb.PROMPTS_PATH)}
    items: list[dict[str, Any]] = []
    for candidate in ("e2", "e5"):
        full = {row["prompt_id"]: row for row in gb.read_jsonl(gb.GEN_DIR / f"{candidate}.jsonl")}
        two = {row["prompt_id"]: row for row in gb.read_jsonl(GEN_DIR / f"{candidate}.two_slot.jsonl")}
        for prompt_id, prompt in prompts.items():
            for mode, rows in (("full", full), ("two_slot", two)):
                items.append(
                    {
                        "group": "condition_availability",
                        "candidate_id": candidate,
                        "condition_mode": mode,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "story": rows[prompt_id]["story"],
                    }
                )
        counterfactual = gb.read_jsonl(GEN_DIR / f"{candidate}.counterfactual.jsonl")
        for row in counterfactual:
            items.append(
                {
                    "group": "counterfactual",
                    "candidate_id": candidate,
                    "condition_mode": "full",
                    "prompt_id": row["prompt_id"],
                    "pair_id": row["pair_id"],
                    "variant": row["variant"],
                    "intervention": row["intervention"],
                    "prompt": row["prompt"],
                    "story": row["story"],
                }
            )
    for row in gb.read_jsonl(gb.GEN_DIR / "e3.jsonl"):
        for mode, story in (("raw", row["raw_story"]), ("repaired", row["story"])):
            items.append(
                {
                    "group": "e3_repair",
                    "candidate_id": "e3",
                    "condition_mode": mode,
                    "prompt_id": row["prompt_id"],
                    "actions": row["actions"],
                    "prompt": row["prompt"],
                    "story": story,
                }
            )
    random.Random(JUDGE_SEED).shuffle(items)
    for index, item in enumerate(items, 1):
        item["blind_id"] = f"AI{index:03d}"
    return items


def pair_items() -> list[dict[str, Any]]:
    prompts = {row["prompt_id"]: row for row in gb.read_jsonl(CF_PROMPTS)}
    items = []
    for candidate in ("e2", "e5"):
        generations = {
            row["prompt_id"]: row
            for row in gb.read_jsonl(GEN_DIR / f"{candidate}.counterfactual.jsonl")
        }
        for pair in counterfactual_pairs():
            a_id, b_id = f"{pair['pair_id']}A", f"{pair['pair_id']}B"
            items.append(
                {
                    "candidate_id": candidate,
                    "pair_id": pair["pair_id"],
                    "intervention": pair["intervention"],
                    "prompt_a": prompts[a_id],
                    "story_a": generations[a_id]["story"],
                    "prompt_b": prompts[b_id],
                    "story_b": generations[b_id]["story"],
                }
            )
    random.Random(JUDGE_SEED + 1).shuffle(items)
    for index, item in enumerate(items, 1):
        item["blind_id"] = f"AP{index:03d}"
    return items


def request_text(prompt: dict[str, Any]) -> str:
    return (
        f"Character: {prompt['character']}\n"
        f"Setting: {prompt['setting']}\n"
        f"Challenge: {prompt['challenge']}\n"
        f"Outcome: {prompt['outcome']}\n"
        f"Teaching: {prompt['teaching']}"
    )


def individual_judge_prompt(item: dict[str, Any]) -> str:
    return (
        "Evaluate the story against the target request. For each *_covered boolean, use true "
        "only when the story unambiguously preserves that specific requested slot; a generic "
        "substitute or merely related motif is false. moral_footer_present is true only when "
        "the final non-empty line begins exactly with 'Moral:'. trait_drives_choice asks whether "
        "the requested character disposition materially affects a decision. choice_causes_outcome "
        "asks whether a character choice produces the resolution. internal_causal_consistency "
        "scores the story's own cause-effect coherence from 1-10, independent of adherence. "
        "requested_causal_consistency scores how well the requested character/challenge/outcome/"
        "teaching form one earned causal chain from 1-10. Return JSON only.\n\n"
        f"TARGET REQUEST:\n{request_text(item['prompt'])}\n\nSTORY:\n{item['story']}"
    )


def pair_judge_prompt(item: dict[str, Any]) -> str:
    intervention_instruction = (
        "Only the requested character trait changes. intervention_changes_decision_or_resolution "
        "is true only if the changed trait materially changes the character's decision or route "
        "through the problem."
        if item["intervention"] == "trait"
        else "Only the requested outcome changes. intervention_changes_decision_or_resolution is "
        "true only if the two stories implement their distinct requested resolutions."
    )
    return (
        "Evaluate this matched counterfactual pair. story_a_matches_a and story_b_matches_b require "
        "faithful implementation of each corresponding request. "
        f"{intervention_instruction} counterfactual_sensitivity is 1-10: 1 means the outputs ignore "
        "the intervention or remain effectively interchangeable; 10 means the changed slot causes "
        "a clear, appropriate change while other held-constant elements remain stable. JSON only.\n\n"
        f"REQUEST A:\n{request_text(item['prompt_a'])}\n\nSTORY A:\n{item['story_a']}\n\n"
        f"REQUEST B:\n{request_text(item['prompt_b'])}\n\nSTORY B:\n{item['story_b']}"
    )


def judge_schema(kind: str) -> dict[str, Any]:
    if kind == "individual":
        properties = {
            field: (
                {"type": "integer", "minimum": 1, "maximum": 10}
                if field.endswith("_consistency")
                else {"type": "boolean"}
            )
            for field in INDIVIDUAL_FIELDS
        }
    else:
        properties = {
            "story_a_matches_a": {"type": "boolean"},
            "story_b_matches_b": {"type": "boolean"},
            "intervention_changes_decision_or_resolution": {"type": "boolean"},
            "counterfactual_sensitivity": {"type": "integer", "minimum": 1, "maximum": 10},
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def judge(kind: str, workers: int) -> None:
    from google import genai
    from google.genai import types

    gb.load_env(ROOT / ".env")
    api_key = os.environ.get("FABLE_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError("FABLE_JUDGE_API_KEY is not configured")
    model_id = os.environ.get("FABLE_JUDGE_MODEL_ID", "gemma-4-26b-a4b-it")
    client = genai.Client(api_key=api_key)
    items = individual_items() if kind == "individual" else pair_items()
    mapping_path = INDIVIDUAL_MAP if kind == "individual" else PAIR_MAP
    score_path = INDIVIDUAL_SCORES if kind == "individual" else PAIR_SCORES
    write_json(
        mapping_path,
        [
            {key: value for key, value in item.items() if key not in {"story", "story_a", "story_b"}}
            for item in items
        ],
    )
    completed = {
        row["blind_id"]
        for row in gb.read_jsonl(score_path)
        if row.get("status") == "ok"
    }
    pending = [item for item in items if item["blind_id"] not in completed]
    print(f"{kind}: {len(completed)} complete, {len(pending)} pending")
    config = types.GenerateContentConfig(
        system_instruction="You are a strict, fair evaluator. Output JSON only.",
        temperature=0,
        seed=JUDGE_SEED,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_json_schema=judge_schema(kind),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL,
            include_thoughts=False,
        ),
    )

    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        prompt = individual_judge_prompt(item) if kind == "individual" else pair_judge_prompt(item)
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                started = time.perf_counter()
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=config,
                )
                raw = response.text or ""
                scores = json.loads(raw)
                if set(scores) != set(judge_schema(kind)["required"]):
                    raise ValueError(f"invalid keys: {scores.keys()}")
                return {
                    "blind_id": item["blind_id"],
                    "status": "ok",
                    "judge_model": model_id,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "scores": scores,
                    "raw": raw,
                }
            except Exception as exc:
                last_error = exc
                if attempt < 8:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"judge failed for {item['blind_id']}: {last_error}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            result = future.result()
            gb.append_jsonl(score_path, result)
            print(f"{item['blind_id']}: {result['latency_ms']} ms", flush=True)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_ROUNDS)
    ]
    return [
        round(percentile(means, 0.025), 3),
        round(percentile(means, 0.975), 3),
    ]


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def exact_moral(story: str, teaching: str) -> bool:
    lines = [line.strip() for line in story.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith("Moral:"):
        return False
    return normalized(lines[-1][len("Moral:") :]) == normalized(teaching)


def attach_scores(kind: str) -> list[dict[str, Any]]:
    mapping_path = INDIVIDUAL_MAP if kind == "individual" else PAIR_MAP
    score_path = INDIVIDUAL_SCORES if kind == "individual" else PAIR_SCORES
    mapping = {row["blind_id"]: row for row in json.loads(mapping_path.read_text())}
    scores = {
        row["blind_id"]: row
        for row in gb.read_jsonl(score_path)
        if row.get("status") == "ok"
    }
    if set(mapping) != set(scores):
        raise ValueError(f"{kind}: mapping/scores mismatch")
    return [{**mapping[blind_id], **scores[blind_id]} for blind_id in mapping]


def metric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row["scores"][field]) for row in rows]


def summarize() -> None:
    individual = attach_scores("individual")
    pairs = attach_scores("pair")
    result: dict[str, Any] = {
        "protocol": json.loads(PROTOCOL.read_text()),
        "condition_availability": {},
        "counterfactual": {},
        "e3_repair": {},
    }
    for candidate in ("e2", "e5"):
        groups = {
            mode: sorted(
                [
                    row
                    for row in individual
                    if row["group"] == "condition_availability"
                    and row["candidate_id"] == candidate
                    and row["condition_mode"] == mode
                ],
                key=lambda row: row["prompt_id"],
            )
            for mode in ("full", "two_slot")
        }
        section: dict[str, Any] = {}
        for mode, rows in groups.items():
            slot_counts = [
                sum(
                    int(row["scores"][f"{slot}_covered"])
                    for slot in ("character", "setting", "challenge", "outcome", "teaching")
                )
                for row in rows
            ]
            provided_counts = [
                int(row["scores"]["character_covered"])
                + int(row["scores"]["teaching_covered"])
                for row in rows
            ]
            added_counts = [
                sum(
                    int(row["scores"][f"{slot}_covered"])
                    for slot in ("setting", "challenge", "outcome")
                )
                for row in rows
            ]
            section[mode] = {
                "n": len(rows),
                "slot_coverage_5_mean": round(statistics.fmean(slot_counts), 3),
                "provided_character_teaching_coverage_2_mean": round(
                    statistics.fmean(provided_counts), 3
                ),
                "setting_challenge_outcome_coverage_3_mean": round(
                    statistics.fmean(added_counts), 3
                ),
                **{
                    field: round(statistics.fmean(metric_values(rows, field)), 3)
                    for field in INDIVIDUAL_FIELDS
                },
            }
        comparisons = {}
        for field in [
            "character_covered",
            "setting_covered",
            "challenge_covered",
            "outcome_covered",
            "teaching_covered",
            "moral_footer_present",
            "trait_drives_choice",
            "choice_causes_outcome",
            "internal_causal_consistency",
            "requested_causal_consistency",
        ]:
            differences = [
                float(full["scores"][field]) - float(two["scores"][field])
                for full, two in zip(groups["full"], groups["two_slot"], strict=True)
            ]
            comparisons[field] = {
                "full_minus_two_slot": round(statistics.fmean(differences), 3),
                "paired_bootstrap_95pct_ci": paired_ci(
                    differences, 20_000 + len(comparisons)
                ),
            }
        slot_differences = [
            sum(
                int(full["scores"][f"{slot}_covered"])
                - int(two["scores"][f"{slot}_covered"])
                for slot in ("character", "setting", "challenge", "outcome", "teaching")
            )
            for full, two in zip(groups["full"], groups["two_slot"], strict=True)
        ]
        comparisons["slot_coverage_5"] = {
            "full_minus_two_slot": round(statistics.fmean(slot_differences), 3),
            "paired_bootstrap_95pct_ci": paired_ci(slot_differences, 20_100),
        }
        section["paired_comparisons"] = comparisons
        result["condition_availability"][candidate] = section

        individual_cf = [
            row
            for row in individual
            if row["group"] == "counterfactual" and row["candidate_id"] == candidate
        ]
        pair_cf = [row for row in pairs if row["candidate_id"] == candidate]
        result["counterfactual"][candidate] = {
            "n_stories": len(individual_cf),
            "n_pairs": len(pair_cf),
            "trait_drives_choice_rate": round(
                statistics.fmean(metric_values(individual_cf, "trait_drives_choice")), 3
            ),
            "choice_causes_outcome_rate": round(
                statistics.fmean(metric_values(individual_cf, "choice_causes_outcome")), 3
            ),
            "requested_causal_consistency": round(
                statistics.fmean(metric_values(individual_cf, "requested_causal_consistency")), 3
            ),
            "pair_match_both_rate": round(
                statistics.fmean(
                    int(row["scores"]["story_a_matches_a"])
                    and int(row["scores"]["story_b_matches_b"])
                    for row in pair_cf
                ),
                3,
            ),
            "pair_match_both_95pct_ci": paired_ci(
                [
                    float(
                        bool(row["scores"]["story_a_matches_a"])
                        and bool(row["scores"]["story_b_matches_b"])
                    )
                    for row in pair_cf
                ],
                25_000 + (0 if candidate == "e2" else 1),
            ),
            "intervention_changes_story_rate": round(
                statistics.fmean(
                    metric_values(pair_cf, "intervention_changes_decision_or_resolution")
                ),
                3,
            ),
            "intervention_changes_story_95pct_ci": paired_ci(
                metric_values(pair_cf, "intervention_changes_decision_or_resolution"),
                25_010 + (0 if candidate == "e2" else 1),
            ),
            "counterfactual_sensitivity": round(
                statistics.fmean(metric_values(pair_cf, "counterfactual_sensitivity")), 3
            ),
            "counterfactual_sensitivity_95pct_ci": paired_ci(
                metric_values(pair_cf, "counterfactual_sensitivity"),
                25_020 + (0 if candidate == "e2" else 1),
            ),
            "by_intervention": {
                intervention: {
                    "n": len(subset),
                    "match_both_rate": round(
                        statistics.fmean(
                            int(row["scores"]["story_a_matches_a"])
                            and int(row["scores"]["story_b_matches_b"])
                            for row in subset
                        ),
                        3,
                    ),
                    "changes_story_rate": round(
                        statistics.fmean(
                            metric_values(
                                subset, "intervention_changes_decision_or_resolution"
                            )
                        ),
                        3,
                    ),
                    "sensitivity": round(
                        statistics.fmean(
                            metric_values(subset, "counterfactual_sensitivity")
                        ),
                        3,
                    ),
                }
                for intervention in ("trait", "outcome")
                if (
                    subset := [
                        row for row in pair_cf if row["intervention"] == intervention
                    ]
                )
            },
        }

    paired_counterfactual = {
        candidate: {
            row["pair_id"]: row
            for row in pairs
            if row["candidate_id"] == candidate
        }
        for candidate in ("e2", "e5")
    }
    result["counterfactual"]["paired_e5_minus_e2"] = {}
    for field in (
        "counterfactual_sensitivity",
        "intervention_changes_decision_or_resolution",
    ):
        differences = [
            float(paired_counterfactual["e5"][pair_id]["scores"][field])
            - float(paired_counterfactual["e2"][pair_id]["scores"][field])
            for pair_id in sorted(paired_counterfactual["e2"])
        ]
        result["counterfactual"]["paired_e5_minus_e2"][field] = {
            "mean_difference": round(statistics.fmean(differences), 3),
            "paired_bootstrap_95pct_ci": paired_ci(
                differences,
                25_100
                + len(result["counterfactual"]["paired_e5_minus_e2"]),
            ),
        }

    e3 = {
        mode: sorted(
            [
                row
                for row in individual
                if row["group"] == "e3_repair" and row["condition_mode"] == mode
            ],
            key=lambda row: row["prompt_id"],
        )
        for mode in ("raw", "repaired")
    }
    e3_generations = {
        row["prompt_id"]: row for row in gb.read_jsonl(gb.GEN_DIR / "e3.jsonl")
    }
    repair_section: dict[str, Any] = {}
    for mode, rows in e3.items():
        stories = [
            (
                e3_generations[row["prompt_id"]]["raw_story"]
                if mode == "raw"
                else e3_generations[row["prompt_id"]]["story"]
            )
            for row in rows
        ]
        repair_section[mode] = {
            "n": len(rows),
            **{
                field: round(statistics.fmean(metric_values(rows, field)), 3)
                for field in INDIVIDUAL_FIELDS
            },
            "exact_requested_moral_rate": round(
                statistics.fmean(
                    exact_moral(story, row["prompt"]["teaching"])
                    for story, row in zip(stories, rows, strict=True)
                ),
                3,
            ),
        }
    repair_section["actions"] = {
        "any_action": sum(bool(row["actions"]) for row in e3_generations.values()),
        "rewrite": sum("rewrite" in row["actions"] for row in e3_generations.values()),
        "moral_postprocess": sum(
            "moral_postprocess" in row["actions"] for row in e3_generations.values()
        ),
    }
    repair_section["paired_comparisons"] = {}
    for field in INDIVIDUAL_FIELDS:
        differences = [
            float(final["scores"][field]) - float(raw["scores"][field])
            for raw, final in zip(e3["raw"], e3["repaired"], strict=True)
        ]
        repair_section["paired_comparisons"][field] = {
            "repaired_minus_raw": round(statistics.fmean(differences), 3),
            "paired_bootstrap_95pct_ci": paired_ci(
                differences, 30_000 + len(repair_section["paired_comparisons"])
            ),
        }
    exact_moral_differences = [
        float(
            exact_moral(
                e3_generations[final["prompt_id"]]["story"],
                final["prompt"]["teaching"],
            )
        )
        - float(
            exact_moral(
                e3_generations[raw["prompt_id"]]["raw_story"],
                raw["prompt"]["teaching"],
            )
        )
        for raw, final in zip(e3["raw"], e3["repaired"], strict=True)
    ]
    repair_section["paired_comparisons"]["exact_requested_moral_rate"] = {
        "repaired_minus_raw": round(statistics.fmean(exact_moral_differences), 3),
        "paired_bootstrap_95pct_ci": paired_ci(exact_moral_differences, 30_100),
    }
    result["e3_repair"] = repair_section
    write_json(SUMMARY, result)

    lines = [
        "# No-retraining ablation summary",
        "",
        "## Condition availability",
        "",
        "| Model | Mode | Slot coverage /5 | Added slots /3 | Requested causal |",
        "|---|---|---:|---:|---:|",
    ]
    for candidate in ("e2", "e5"):
        for mode in ("full", "two_slot"):
            row = result["condition_availability"][candidate][mode]
            lines.append(
                f"| {candidate} | {mode} | {row['slot_coverage_5_mean']:.2f} | "
                f"{row['setting_challenge_outcome_coverage_3_mean']:.2f} | "
                f"{row['requested_causal_consistency']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Counterfactual sensitivity",
            "",
            "| Model | Match both | Intervention changes story | Sensitivity /10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate in ("e2", "e5"):
        row = result["counterfactual"][candidate]
        lines.append(
            f"| {candidate} | {row['pair_match_both_rate']:.2f} | "
            f"{row['intervention_changes_story_rate']:.2f} | "
            f"{row['counterfactual_sensitivity']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## E4 repair",
            "",
            "| Mode | Exact moral | Moral footer | Requested causal |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in ("raw", "repaired"):
        row = result["e3_repair"][mode]
        lines.append(
            f"| {mode} | {row['exact_requested_moral_rate']:.2f} | "
            f"{row['moral_footer_present']:.2f} | "
            f"{row['requested_causal_consistency']:.2f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_files = [
        PROTOCOL,
        CF_PROMPTS,
        *(GEN_DIR / f"{candidate}.{suite}.jsonl"
          for candidate in ("e2", "e5")
          for suite in ("two_slot", "counterfactual")),
        INDIVIDUAL_MAP,
        INDIVIDUAL_SCORES,
        PAIR_MAP,
        PAIR_SCORES,
        SUMMARY,
        SUMMARY_MD,
        gb.GEN_DIR / "e2.jsonl",
        gb.GEN_DIR / "e3.jsonl",
        gb.GEN_DIR / "e5.jsonl",
        Path(__file__),
    ]
    write_json(
        RUN_MANIFEST,
        {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": sys.version.split()[0],
            "files": {
                str(path.relative_to(ROOT)): {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in manifest_files
            },
        },
    )
    print("\n".join(lines))


def validate() -> None:
    cf = gb.read_jsonl(CF_PROMPTS)
    if len(cf) != 20 or len({row["prompt_id"] for row in cf}) != 20:
        raise ValueError("expected 20 unique counterfactual prompts")
    for candidate in ("e2", "e5"):
        for suite, expected in (("two_slot", 25), ("counterfactual", 20)):
            rows = gb.read_jsonl(GEN_DIR / f"{candidate}.{suite}.jsonl")
            if len(rows) != expected or len({row["prompt_id"] for row in rows}) != expected:
                raise ValueError(f"{candidate}/{suite}: expected {expected} unique rows")
    if INDIVIDUAL_SCORES.exists():
        individual = gb.read_jsonl(INDIVIDUAL_SCORES)
        if len(individual) != 190 or len({row["blind_id"] for row in individual}) != 190:
            raise ValueError("expected 190 individual judgments")
    if PAIR_SCORES.exists():
        pairs = gb.read_jsonl(PAIR_SCORES)
        if len(pairs) != 20 or len({row["blind_id"] for row in pairs}) != 20:
            raise ValueError("expected 20 pair judgments")
    print("OK: ablation prompts, generations, and judgments")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--candidate", choices=["e2", "e5"], required=True)
    generate_parser.add_argument(
        "--suite", choices=["two_slot", "counterfactual"], required=True
    )
    judge_parser = sub.add_parser("judge")
    judge_parser.add_argument("--kind", choices=["individual", "pair"], required=True)
    judge_parser.add_argument("--workers", type=int, default=6)
    sub.add_parser("summarize")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "generate":
        generate(args.candidate, args.suite)
    elif args.command == "judge":
        judge(args.kind, args.workers)
    elif args.command == "summarize":
        summarize()
    else:
        validate()


if __name__ == "__main__":
    main()
