"""Extend the failure-focused LoRA dataset with TF1 corrected targets.

The seed dataset comes from observed Base failures. This script adds curated
TF1 examples as corrected targets for the same failure taxonomy so the LoRA run
has enough rows to train.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

from scripts.build_failure_focused_lora import SYSTEM, llama32_chat_text


FAILURE_PATTERNS = [
    ["missing_or_empty_moral", "weak_moral_alignment"],
    ["unclean_ending", "weak_moral_alignment"],
    ["weak_outcome_coverage", "weak_moral_alignment"],
    ["empty_moral", "missing_or_empty_moral"],
    ["missing_or_empty_moral", "unclean_ending"],
]


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


def has_clean_target(row: dict) -> bool:
    output = row.get("output", "").strip()
    input_text = row.get("input", "")
    return (
        len(output.split()) >= 80
        and len(output.split()) <= 380
        and "Moral:" in output
        and "Character:" in input_text
        and "Teaching:" in input_text
    )


def punctuate(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def user_text(row: dict) -> str:
    return f"{row['instruction'].strip()}\n\n{row['input'].strip()}"


def convert_tf1_row(row: dict, index: int) -> dict:
    user = user_text(row)
    output = punctuate(row["output"])
    return {
        "id": f"tf1_failure_aug_{index:04d}",
        "instruction": row["instruction"],
        "input": row["input"],
        "output": output,
        "failure_reasons": FAILURE_PATTERNS[index % len(FAILURE_PATTERNS)],
        "source_model_id": "tf1_augmented_failure_target",
        "repair_model_id": "gold_target",
        "text": llama32_chat_text(SYSTEM, user, output),
    }


def reason_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row.get("failure_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(source_dir).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", default="data/failure_focused_lora")
    parser.add_argument("--tf1-train", default="data/tf1/sft_10000/train.jsonl")
    parser.add_argument("--target-total", type=int, default=300)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", default="data/failure_focused_lora_300")
    parser.add_argument("--zip", default="data/failure_focused_lora_300.zip")
    parser.add_argument("--seed", type=int, default=5410)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seed_rows = read_jsonl(Path(args.seed_dir) / "all.jsonl")
    tf1_rows = [row for row in read_jsonl(Path(args.tf1_train)) if has_clean_target(row)]
    rng.shuffle(tf1_rows)

    needed = max(0, args.target_total - len(seed_rows))
    augmented = [convert_tf1_row(row, i) for i, row in enumerate(tf1_rows[:needed], 1)]
    all_rows = seed_rows + augmented
    rng.shuffle(all_rows)

    valid_size = max(1, round(len(all_rows) * args.valid_ratio))
    valid_rows = all_rows[:valid_size]
    train_rows = all_rows[valid_size:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "all.jsonl", all_rows)
    manifest = {
        "name": "failure_focused_lora_300",
        "purpose": "Extended failure-focused LoRA dataset.",
        "total": len(all_rows),
        "seed_failure_rows": len(seed_rows),
        "tf1_augmented_rows": len(augmented),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "reason_counts": reason_counts(all_rows),
        "method": (
            "Observed Base failure rows are kept. Additional TF1 corrected targets "
            "are sampled as gold outputs and tagged with the same failure taxonomy "
            "to train the model toward complete moral-aligned fables."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_dir(out_dir, Path(args.zip))
    print(f"Wrote {out_dir}")
    print(f"Wrote {args.zip}")
    print(f"Rows: total={len(all_rows)} train={len(train_rows)} valid={len(valid_rows)}")


if __name__ == "__main__":
    main()
