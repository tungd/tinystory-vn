"""Quality-weighted continuation from immutable v3-full weights."""

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_v3 import encode_story_only, validate_output_path


def resolve_max_steps(train_examples: int, batch: int, epochs: int, requested: int) -> int:
    if train_examples < 1 or batch < 1 or epochs < 1 or requested < 0:
        raise ValueError("invalid v5 training size")
    return requested or math.ceil(train_examples / batch) * epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runs/v5/data/prepared")
    parser.add_argument("--base-model", default="runs/v3/artifacts/hf")
    parser.add_argument("--out", default="runs/v5/artifacts/hf")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    data_dir = Path(args.data)
    base_config = Path(args.base_model) / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(f"Missing v3-full model config: {base_config}")
    raw_tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    eos_id = raw_tokenizer.token_to_id("</story>")
    if eos_id is None:
        raise ValueError("Tokenizer lacks </story> token")

    datasets = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
    })
    if args.validation_samples:
        datasets["validation"] = datasets["validation"].select(
            range(min(args.validation_samples, len(datasets["validation"])))
        )

    def encode(example):
        return encode_story_only(
            raw_tokenizer, example["prompt"], example["target"], args.block_size, eos_id
        )

    encoded = datasets.map(encode, remove_columns=datasets["train"].column_names)
    first = encoded["train"][0]
    masked = sum(label == -100 for label in first["labels"])
    if masked == 0 or masked == len(first["labels"]):
        raise ValueError("Expected masked prompt and unmasked target tokens")
    max_steps = resolve_max_steps(len(encoded["train"]), args.batch, args.epochs, args.max_steps)
    resolved = {
        **vars(args),
        "max_steps": max_steps,
        "train_examples": len(encoded["train"]),
        "validation_examples": len(encoded["validation"]),
    }
    if args.dry_run:
        print(json.dumps({
            "status": "ready",
            "base_model": args.base_model,
            "train_examples": len(encoded["train"]),
            "validation_examples": len(encoded["validation"]),
            "max_steps": max_steps,
            "first_tokens": len(first["input_ids"]),
            "first_masked_prompt_tokens": masked,
            "output": args.out,
        }, indent=2))
        return

    output = validate_output_path(args.base_model, args.out)
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    tokenizer.pad_token = "</story>"
    tokenizer.eos_token = "</story>"
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("v3-full model/tokenizer vocabulary mismatch")

    class StoryOnlyCollator:
        def __call__(self, features):
            max_length = max(len(feature["input_ids"]) for feature in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                padding = max_length - len(feature["input_ids"])
                batch["input_ids"].append(
                    feature["input_ids"] + [tokenizer.pad_token_id] * padding
                )
                batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
                batch["labels"].append(feature["labels"] + [-100] * padding)
            return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=args.batch,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=min(args.warmup_steps, max_steps // 10),
        lr_scheduler_type="cosine",
        logging_steps=10,
        report_to="none",
        eval_strategy="steps",
        eval_steps=min(args.save_steps, max_steps),
        per_device_eval_batch_size=args.batch,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=min(args.save_steps, max_steps),
        save_total_limit=None,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=StoryOnlyCollator(),
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "v5_training_config.json").write_text(
        json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
