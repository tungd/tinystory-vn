"""Run the baseline model on the fixed evaluation prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import GEN_REPEAT_PENALTY, GEN_TEMPERATURE, GEN_TOP_P
from app.guardrail.output_filter import check_output_en
from app.models_registry import resolve_ollama
from app.ollama_client import OllamaError, generate_meta
from app.prompt_en import (
    LENGTH_HINT_EN,
    LENGTH_NUM_PREDICT,
    SYSTEM_PROMPT_EN,
    build_fable_prompt,
)


DEFAULT_MODEL_ID = "base-llama32-3b-instruct"
DEFAULT_SEED = 5410


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_summary(row: dict) -> str:
    return (
        f"Character: {row['character']}\n"
        f"Setting: {row['setting']}\n"
        f"Challenge: {row['challenge']}\n"
        f"Outcome: {row['outcome']}\n"
        f"Teaching: {row['teaching']}\n"
        f"Length: {row['length']}"
    )


def write_markdown(path: Path, rows: list[dict], model_id: str, ollama_model: str) -> None:
    lines = [
        "# Baseline Outputs",
        "",
        f"- Model ID: `{model_id}`",
        f"- Ollama model: `{ollama_model}`",
        f"- Total prompts: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                "### Prompt",
                "",
                "```text",
                prompt_summary(row["input"]),
                "```",
                "",
                "### Output",
                "",
                row.get("story", "").strip() or "_No story generated._",
                "",
                "### Metadata",
                "",
                f"- Status: `{row['status']}`",
                f"- Output safety: `{row.get('output_safety', 'not_checked')}`",
                f"- Input tokens: {row.get('meta', {}).get('input_tokens', 0)}",
                f"- Output tokens: {row.get('meta', {}).get('output_tokens', 0)}",
                f"- Latency ms: {row.get('meta', {}).get('latency_ms', 0)}",
                "",
            ]
        )
        if row.get("error"):
            lines.extend(["### Error", "", row["error"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    prompts = read_jsonl(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    ollama_model = resolve_ollama(args.model_id)
    outputs: list[dict] = []

    for index, row in enumerate(prompts, 1):
        length = row.get("length", "medium")
        prompt = build_fable_prompt(
            row.get("character", ""),
            row.get("setting", ""),
            row.get("challenge", ""),
            row.get("outcome", ""),
            row.get("teaching", ""),
            LENGTH_HINT_EN[length],
        )
        print(f"[{index}/{len(prompts)}] {row['id']} -> {ollama_model}", flush=True)
        try:
            result = generate_meta(
                prompt=prompt,
                system=SYSTEM_PROMPT_EN,
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
                    "model_id": args.model_id,
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
                    "model_id": args.model_id,
                    "ollama": ollama_model,
                    "status": "error",
                    "input": row,
                    "prompt_sent": prompt,
                    "story": "",
                    "error": str(exc),
                }
            )

    jsonl_path = Path(args.out_jsonl)
    md_path = Path(args.out_md)
    write_jsonl(jsonl_path, outputs)
    write_markdown(md_path, outputs, args.model_id, ollama_model)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/test_prompts.jsonl")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-jsonl", default="results/baseline_outputs.jsonl")
    parser.add_argument("--out-md", default="results/baseline_outputs.md")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
