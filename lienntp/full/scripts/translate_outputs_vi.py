"""Translate generated fable outputs to Vietnamese for quick quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ollama_client import generate_meta


SYSTEM_PROMPT = (
    "You are a careful English to Vietnamese translator for children's fables. "
    "Translate naturally into Vietnamese, keep the meaning, preserve all animal species, "
    "character names, events, and order of events, keep the story wholesome, "
    "do not add new details, and make the final moral line start exactly with 'Bài học:'. "
    "Return only the Vietnamese translation."
)


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


def prompt_summary(prompt: dict) -> str:
    return (
        f"Nhân vật: {prompt.get('character', '')}\n"
        f"Bối cảnh: {prompt.get('setting', '')}\n"
        f"Thử thách: {prompt.get('challenge', '')}\n"
        f"Kết quả: {prompt.get('outcome', '')}\n"
        f"Bài học: {prompt.get('teaching', '')}\n"
        f"Độ dài: {prompt.get('length', '')}"
    )


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = ["# Vietnamese Translation Samples", ""]
    for row in rows:
        prompt = row.get("input", {})
        lines.extend(
            [
                f"## {row['id']}",
                "",
                "### Prompt",
                "",
                "```text",
                prompt_summary(prompt),
                "```",
                "",
                "### English",
                "",
                row.get("story", "").strip() or "_No story._",
                "",
                "### Vietnamese",
                "",
                row.get("story_vi", "").strip() or "_No translation._",
                "",
                "### Metadata",
                "",
                f"- Translation model: `{row.get('translation_model', '')}`",
                f"- Translation latency ms: {row.get('translation_meta', {}).get('latency_ms', 0)}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def translate_story(story: str, model: str, num_predict: int) -> dict:
    prompt = (
        "Translate this English children's fable into Vietnamese. "
        "Do not summarize. Preserve the narrative and translate the moral as a final line.\n\n"
        f"{story}"
    )
    return generate_meta(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        model=model,
        num_predict=num_predict,
        seed=42,
        temperature=0.0,
        top_p=0.9,
        repeat_penalty=1.05,
    )


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]

    translated: list[dict] = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] Translating {row['id']} with {args.model}", flush=True)
        result = translate_story(row.get("story", ""), args.model, args.num_predict)
        output = dict(row)
        output["story_vi"] = result["text"].strip()
        output["translation_model"] = args.model
        output["translation_meta"] = {
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "latency_ms": result.get("latency_ms", 0),
        }
        translated.append(output)

    write_jsonl(Path(args.out_jsonl), translated)
    write_markdown(Path(args.out_md), translated)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_md}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/sft_llama32_fable_10k_outputs.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--num-predict", type=int, default=900)
    parser.add_argument("--out-jsonl", default="results/sft10k_vi_translation_sample.jsonl")
    parser.add_argument("--out-md", default="results/sft10k_vi_translation_sample.md")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
