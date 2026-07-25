"""Run Base + validation/post-processing on fixed evaluation prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import GEN_REPEAT_PENALTY, GEN_TEMPERATURE, GEN_TOP_P
from app.enhanced_generation import enhance_story
from app.guardrail.output_filter import check_output_en
from app.models_registry import resolve_ollama
from app.ollama_client import OllamaError, generate_meta
from app.prompt_en import (
    LENGTH_HINT_EN,
    LENGTH_NUM_PREDICT,
    SYSTEM_PROMPT_EN,
    build_fable_prompt,
)
from scripts.run_baseline import prompt_summary, write_markdown


DEFAULT_MODEL_ID = "base-llama32-3b-instruct"
DEFAULT_SEED = 5410


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_enhanced_markdown(path: Path, rows: list[dict], model_id: str, ollama_model: str) -> None:
    lines = [
        "# Enhanced Base Outputs",
        "",
        f"- Model ID: `{model_id}`",
        f"- Ollama model: `{ollama_model}`",
        f"- Total prompts: {len(rows)}",
        "",
    ]
    for row in rows:
        enhanced = row.get("enhancement", {})
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
                "### Enhancement",
                "",
                f"- Actions: `{', '.join(enhanced.get('actions', [])) or 'none'}`",
                f"- Initial fixable: `{', '.join(enhanced.get('initial_validation', {}).get('fixable_reasons', [])) or 'none'}`",
                f"- Initial severe: `{', '.join(enhanced.get('initial_validation', {}).get('severe_reasons', [])) or 'none'}`",
                f"- Final fixable: `{', '.join(enhanced.get('final_validation', {}).get('fixable_reasons', [])) or 'none'}`",
                f"- Final severe: `{', '.join(enhanced.get('final_validation', {}).get('severe_reasons', [])) or 'none'}`",
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
        if row.get("raw_story"):
            lines.extend(["### Raw Output", "", row["raw_story"].strip(), ""])
        if row.get("error"):
            lines.extend(["### Error", "", row["error"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_raw(row: dict, model: str, seed: int | None) -> dict:
    length = row.get("length", "medium")
    prompt = build_fable_prompt(
        row.get("character", ""),
        row.get("setting", ""),
        row.get("challenge", ""),
        row.get("outcome", ""),
        row.get("teaching", ""),
        LENGTH_HINT_EN[length],
    )
    result = generate_meta(
        prompt=prompt,
        system=SYSTEM_PROMPT_EN,
        model=model,
        num_predict=LENGTH_NUM_PREDICT[length],
        seed=seed,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        repeat_penalty=GEN_REPEAT_PENALTY,
    )
    return {
        "story": result["text"],
        "prompt_sent": prompt,
        "meta": {
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "latency_ms": result.get("latency_ms", 0),
        },
    }


def source_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def run(args: argparse.Namespace) -> int:
    prompts = read_jsonl(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    model = resolve_ollama(args.model_id)
    rewrite_model = resolve_ollama(args.rewrite_model_id) if args.rewrite else None
    source_rows = source_by_id(read_jsonl(Path(args.source_jsonl))) if args.source_jsonl else {}
    outputs: list[dict] = []

    for index, row in enumerate(prompts, 1):
        print(f"[{index}/{len(prompts)}] {row['id']} -> enhanced {model}", flush=True)
        try:
            source = source_rows.get(row["id"])
            if source:
                raw_story = source.get("story", "")
                raw_meta = source.get("meta", {})
                prompt_sent = source.get("prompt_sent", "")
            else:
                generated = generate_raw(row, model, args.seed)
                raw_story = generated["story"]
                raw_meta = generated["meta"]
                prompt_sent = generated["prompt_sent"]

            enhancement = enhance_story(
                raw_story,
                row,
                rewrite_model=rewrite_model,
                generate_meta_fn=generate_meta if args.rewrite else None,
                seed=args.seed,
            )
            story = enhancement["story"]
            safety = check_output_en(story)
            latency_ms = raw_meta.get("latency_ms", 0) + enhancement.get("extra_latency_ms", 0)
            outputs.append(
                {
                    "id": row["id"],
                    "model_id": args.output_model_id,
                    "ollama": model,
                    "status": "success",
                    "input": row,
                    "prompt_sent": prompt_sent,
                    "raw_story": raw_story,
                    "story": story,
                    "enhancement": enhancement,
                    "output_safety": "ok" if safety.ok else "blocked",
                    "output_safety_reason": safety.reason,
                    "meta": {
                        "input_tokens": raw_meta.get("input_tokens", 0),
                        "output_tokens": raw_meta.get("output_tokens", 0),
                        "latency_ms": latency_ms,
                    },
                }
            )
        except (KeyError, OllamaError) as exc:
            outputs.append(
                {
                    "id": row["id"],
                    "model_id": args.output_model_id,
                    "ollama": model,
                    "status": "error",
                    "input": row,
                    "story": "",
                    "error": str(exc),
                }
            )

    write_jsonl(Path(args.out_jsonl), outputs)
    write_enhanced_markdown(Path(args.out_md), outputs, args.output_model_id, model)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_md}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/test_prompts.jsonl")
    parser.add_argument("--source-jsonl", default="results/baseline_outputs.jsonl")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-model-id", default="base-llama32-3b-instruct-enhanced")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-jsonl", default="results/base_enhanced_outputs.jsonl")
    parser.add_argument("--out-md", default="results/base_enhanced_outputs.md")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
