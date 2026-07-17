"""Diagnose V5 memorization, length sensitivity, and moral conditioning."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_v3 import PREFIX


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_human_controls(rows: list[dict], count: int, seed: int) -> list[dict]:
    unique = {}
    for row in rows:
        if row.get("source_type") == "human-authored":
            unique[row["source"]] = row
    ordered = [unique[source] for source in sorted(unique)]
    if len(ordered) < count:
        raise ValueError(f"Need {count} unique human rows; found {len(ordered)}")
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda row: row["source"])


def base_control(row: dict, split: str) -> dict:
    return {
        "source": row["source"],
        "split": split,
        "character": row["character"],
        "original_moral": row["moral"],
        "reference_story": row["target"].rsplit("\n\nMoral:", 1)[0],
    }


def diagnostic_groups(
    train_rows: list[dict],
    validation_rows: list[dict],
    *,
    count: int = 20,
    lengths: tuple[int, ...] = (120, 180, 300),
    ablation_length: int = 180,
    seed: int = 42,
) -> list[dict]:
    groups = []
    for split, rows, split_seed in (
        ("train", train_rows, seed),
        ("holdout", validation_rows, seed + 1),
    ):
        selected = [
            base_control(row, split)
            for row in select_human_controls(rows, count, split_seed)
        ]
        swapped = [row["original_moral"] for row in selected[1:]] + [
            selected[0]["original_moral"]
        ]
        for length in lengths:
            controls = []
            for row in selected:
                moral = row["original_moral"]
                controls.append({
                    **row,
                    "condition": "original",
                    "requested_moral": moral,
                    "prompt": PREFIX.format(character=row["character"], moral=moral),
                })
            groups.append({"split": split, "condition": "original", "max_new_tokens": length, "controls": controls})
        for condition, morals in (("swapped_moral", swapped), ("blank_moral", [""] * len(selected))):
            controls = []
            for row, moral in zip(selected, morals):
                controls.append({
                    **row,
                    "condition": condition,
                    "requested_moral": moral,
                    "prompt": PREFIX.format(character=row["character"], moral=moral),
                })
            groups.append({
                "split": split,
                "condition": condition,
                "max_new_tokens": ablation_length,
                "controls": controls,
            })
    return groups


def generate_group(model, tokenizer, group: dict, batch: int) -> list[dict]:
    import torch

    controls = group["controls"]
    rows = []
    for start in range(0, len(controls), batch):
        batch_controls = controls[start : start + batch]
        encoded = tokenizer(
            [row["prompt"] for row in batch_controls],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=group["max_new_tokens"],
                do_sample=False,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[:, encoded["input_ids"].shape[1] :]
        for control, tokens in zip(batch_controls, generated):
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            rows.append({
                **control,
                "max_new_tokens": group["max_new_tokens"],
                "output_tokens": int(tokens.shape[0]),
                "ended": "</story>" in text,
                "story": text.split("</story>", 1)[0].strip(),
            })
    print(
        f"{group['split']} {group['condition']} {group['max_new_tokens']}: {len(rows)}",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="runs/v5/data/prepared")
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--lengths", type=int, nargs="+", default=[120, 180, 300])
    parser.add_argument("--ablation-length", type=int, default=180)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    data = Path(args.data)
    groups = diagnostic_groups(
        load_jsonl(data / "train.jsonl"),
        load_jsonl(data / "validation.jsonl"),
        count=args.controls,
        lengths=tuple(args.lengths),
        ablation_length=args.ablation_length,
        seed=args.seed,
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    tokenizer.pad_token = "</story>"
    tokenizer.eos_token = "</story>"
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda").eval()
    rows = []
    for group in groups:
        rows.extend(generate_group(model, tokenizer, group, args.batch))
    del model
    gc.collect()

    result = {
        "kind": "v5-conditioning-diagnostic",
        "model": args.model,
        "settings": {
            "controls_per_split": args.controls,
            "lengths": args.lengths,
            "ablation_length": args.ablation_length,
            "decoding": "greedy",
            "repetition_penalty": 1.3,
            "seed": args.seed,
        },
        "generations": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
