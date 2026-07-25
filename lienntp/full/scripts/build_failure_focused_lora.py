"""Build a failure-focused SFT dataset for LoRA training.

The dataset teaches the model to produce the corrected story directly from the
original fable prompt. It keeps only examples where the base model showed a
detected weakness or where the repaired output materially differs from base.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path


SYSTEM = (
    "You are a storyteller who writes short fables for young children. "
    "Write coherent English fables with a clear final Moral line."
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_text(prompt: dict) -> str:
    return (
        "Write a short English fable for children with a clear moral.\n\n"
        f"Character: {prompt.get('character', '')}\n"
        f"Setting: {prompt.get('setting', '')}\n"
        f"Challenge: {prompt.get('challenge', '')}\n"
        f"Outcome: {prompt.get('outcome', '')}\n"
        f"Teaching: {prompt.get('teaching', '')}\n"
        f"Length: {prompt.get('length', 'short')}"
    )


def llama32_chat_text(system: str, user: str, assistant: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant.strip()}<|eot_id|>"
    )


def collect_failure_reasons(base_eval: dict, repair_row: dict) -> list[str]:
    reasons: list[str] = []
    if not base_eval.get("has_moral_footer"):
        reasons.append("missing_or_empty_moral")
    if base_eval.get("empty_moral"):
        reasons.append("empty_moral")
    if not base_eval.get("clean_ending"):
        reasons.append("unclean_ending")
    if not base_eval.get("moral_exact"):
        reasons.append("weak_moral_alignment")
    if not base_eval.get("outcome_covered"):
        reasons.append("weak_outcome_coverage")
    if base_eval.get("has_run_on_sentence"):
        reasons.append("run_on_sentence")
    enhancement = repair_row.get("enhancement", {})
    actions = enhancement.get("actions", [])
    if actions:
        reasons.extend(f"action_{action}" for action in actions)
    return sorted(set(reasons))


def build_rows(base_outputs: list[dict], repair_outputs: list[dict], base_eval: dict) -> list[dict]:
    repair_by_id = {row["id"]: row for row in repair_outputs}
    eval_by_id = {row["id"]: row for row in base_eval["rows"]}
    rows: list[dict] = []

    for base in base_outputs:
        if base.get("status") != "success":
            continue
        rid = base["id"]
        repair = repair_by_id.get(rid)
        metrics = eval_by_id.get(rid)
        if not repair or repair.get("status") != "success" or not metrics:
            continue

        reasons = collect_failure_reasons(metrics, repair)
        changed = base.get("story", "").strip() != repair.get("story", "").strip()
        if not reasons and not changed:
            continue

        user = prompt_text(base["input"])
        assistant = repair["story"].strip()
        rows.append(
            {
                "id": rid,
                "instruction": "Write a short English fable for children with a clear moral.",
                "input": user.split("\n\n", 1)[1],
                "output": assistant,
                "failure_reasons": reasons,
                "source_model_id": base.get("model_id", ""),
                "repair_model_id": repair.get("model_id", ""),
                "text": llama32_chat_text(SYSTEM, user, assistant),
            }
        )

    return rows


def write_manifest(path: Path, rows: list[dict], train_rows: list[dict], valid_rows: list[dict]) -> None:
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["failure_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    manifest = {
        "name": "failure_focused_lora_seed",
        "purpose": "Small seed dataset for failure-focused LoRA training.",
        "total": len(rows),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "note": (
            "This is a seed dataset generated from the fixed 25-prompt benchmark. "
            "For a real LoRA run, extend it to at least 100-300 examples by running "
            "more prompts through the same Base -> Repair pipeline."
        ),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(source_dir).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-outputs", default="results/baseline_outputs.jsonl")
    parser.add_argument("--repair-outputs", default="results/base_repair_outputs.jsonl")
    parser.add_argument("--base-eval", default="results/eval_base.json")
    parser.add_argument("--out-dir", default="data/failure_focused_lora")
    parser.add_argument("--zip", default="data/failure_focused_lora.zip")
    parser.add_argument("--seed", type=int, default=5410)
    args = parser.parse_args()

    rows = build_rows(
        read_jsonl(Path(args.base_outputs)),
        read_jsonl(Path(args.repair_outputs)),
        json.loads(Path(args.base_eval).read_text(encoding="utf-8")),
    )
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    valid_size = max(1, round(len(rows) * 0.2)) if len(rows) > 1 else 0
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "all.jsonl", rows)
    write_manifest(out_dir / "manifest.json", rows, train_rows, valid_rows)
    zip_dir(out_dir, Path(args.zip))
    print(f"Wrote {out_dir}")
    print(f"Wrote {args.zip}")
    print(f"Rows: total={len(rows)} train={len(train_rows)} valid={len(valid_rows)}")


if __name__ == "__main__":
    main()
