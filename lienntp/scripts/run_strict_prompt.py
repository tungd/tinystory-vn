"""Run a strict-prompt ablation on the fixed English fable benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import GEN_REPEAT_PENALTY, GEN_TEMPERATURE, GEN_TOP_P
from app.guardrail.output_filter import check_output_en
from app.models_registry import resolve_ollama
from app.ollama_client import OllamaError, generate_meta
from app.prompt_en import LENGTH_NUM_PREDICT
from scripts.run_baseline import prompt_summary, write_jsonl, write_markdown


STRICT_SYSTEM_PROMPT = (
    "You are a precise children's fable writer. "
    "Follow the user's requested character, setting, challenge, outcome, and teaching. "
    "Write simple, fluent English for ages 4-7. "
    "Never write bullet points. Never leave the moral blank. "
    "The final line must be exactly one line starting with 'Moral:'."
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def strict_length_hint(length: str) -> str:
    if length == "short":
        return "Write 120-180 words total."
    if length == "long":
        return "Write 420-550 words total."
    return "Write 230-330 words total."


def punctuate(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def build_strict_prompt(row: dict) -> str:
    teaching = punctuate(row.get("teaching", ""))
    return (
        "Write exactly one complete English children's fable.\n"
        "Required structure:\n"
        "1. Paragraph 1: introduce the requested character and setting.\n"
        "2. Paragraph 2: show the challenge and the outcome clearly.\n"
        "3. Final line: write only the moral.\n\n"
        "Hard constraints:\n"
        f"- Use this main character explicitly: {row.get('character', '')}\n"
        f"- Use this setting explicitly: {row.get('setting', '')}\n"
        f"- Include this challenge: {row.get('challenge', '')}\n"
        f"- Include this outcome: {row.get('outcome', '')}\n"
        f"- The final line must be exactly: Moral: {teaching}\n"
        "- Do not continue after the moral line.\n"
        "- Do not use lists, headings, notes, or explanations.\n"
        f"- {strict_length_hint(row.get('length', 'medium'))}"
    )


def run(args: argparse.Namespace) -> int:
    prompts = read_jsonl(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    ollama_model = resolve_ollama(args.model_id)
    outputs: list[dict] = []

    for index, row in enumerate(prompts, 1):
        length = row.get("length", "medium")
        prompt = build_strict_prompt(row)
        print(f"[{index}/{len(prompts)}] {row['id']} -> strict {ollama_model}", flush=True)
        try:
            result = generate_meta(
                prompt=prompt,
                system=STRICT_SYSTEM_PROMPT,
                model=ollama_model,
                num_predict=LENGTH_NUM_PREDICT[length],
                seed=args.seed,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                repeat_penalty=GEN_REPEAT_PENALTY,
            )
            safety = check_output_en(result["text"])
            outputs.append(
                {
                    "id": row["id"],
                    "model_id": args.output_model_id,
                    "ollama": ollama_model,
                    "status": "success",
                    "input": row,
                    "prompt_sent": prompt,
                    "story": result["text"],
                    "output_safety": "ok" if safety.ok else "blocked",
                    "output_safety_reason": safety.reason,
                    "meta": {
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "latency_ms": result.get("latency_ms", 0),
                    },
                }
            )
        except (KeyError, OllamaError) as exc:
            outputs.append(
                {
                    "id": row["id"],
                    "model_id": args.output_model_id,
                    "ollama": ollama_model,
                    "status": "error",
                    "input": row,
                    "prompt_sent": prompt,
                    "story": "",
                    "error": str(exc),
                }
            )

    write_jsonl(Path(args.out_jsonl), outputs)
    write_markdown(Path(args.out_md), outputs, args.output_model_id, ollama_model)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_md}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/test_prompts.jsonl")
    parser.add_argument("--model-id", default="base-llama32-3b-instruct-q4")
    parser.add_argument("--output-model-id", default="base-llama32-3b-strict-prompt")
    parser.add_argument("--seed", type=int, default=5410)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-jsonl", default="results/strict_prompt_outputs.jsonl")
    parser.add_argument("--out-md", default="results/strict_prompt_outputs.md")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
