"""Continue the from-scratch v2 model on v3 targets with prompt loss masked."""

import argparse
import json
from pathlib import Path


def encode_story_only(tokenizer, prompt: str, target: str, block_size: int, eos_id: int) -> dict:
    """Encode one example; only target tokens contribute to causal-LM loss."""
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(target).ids
    room = block_size - len(prompt_ids)
    if room <= 0:
        raise ValueError("Control prefix exceeds block size")
    if len(target_ids) > room:
        target_ids = target_ids[:room]
        target_ids[-1] = eos_id
    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def validate_output_path(base_model: str | Path, output: str | Path) -> Path:
    """Keep v3 writes isolated from v2 and refuse accidental reruns."""
    base = Path(base_model).resolve()
    out = Path(output).resolve()
    if out == base or out.is_relative_to(base) or base.is_relative_to(out):
        raise ValueError("v3 output must be isolated from the v2 model directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing non-empty v3 output directory: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runs/v3/data")
    parser.add_argument("--base-model", default="runs/v2/artifacts/hf")
    parser.add_argument("--out", default="runs/v3/artifacts/hf")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=313)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--train-samples", type=int, default=20_000)
    parser.add_argument("--validation-samples", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    data_dir = Path(args.data)
    base_config = Path(args.base_model) / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(f"Missing v2 model config: {base_config}")
    raw_tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    eos_id = raw_tokenizer.token_to_id("</story>")
    if eos_id is None:
        raise ValueError("Tokenizer lacks </story> token")

    datasets = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "validation.jsonl"),
        },
    )
    if args.train_samples:
        count = min(args.train_samples, len(datasets["train"]))
        datasets["train"] = datasets["train"].select(range(count))
    if args.validation_samples:
        count = min(args.validation_samples, len(datasets["validation"]))
        datasets["validation"] = datasets["validation"].select(range(count))

    def encode(example):
        return encode_story_only(
            raw_tokenizer,
            example["prompt"],
            example["target"],
            args.block_size,
            eos_id,
        )

    encoded = datasets.map(encode, remove_columns=datasets["train"].column_names)
    first = encoded["train"][0]
    masked = sum(label == -100 for label in first["labels"])
    if masked == 0 or masked == len(first["labels"]):
        raise ValueError("Expected masked prompt and unmasked target tokens")

    if args.dry_run:
        print(json.dumps({
            "status": "ready",
            "base_model": args.base_model,
            "train_examples": len(encoded["train"]),
            "validation_examples": len(encoded["validation"]),
            "first_tokens": len(first["input_ids"]),
            "first_masked_prompt_tokens": masked,
            "output": args.out,
        }, indent=2))
        return

    output = validate_output_path(args.base_model, args.out)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    tokenizer.pad_token = "</story>"
    tokenizer.eos_token = "</story>"
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("v2 model/tokenizer vocabulary mismatch")

    class StoryOnlyCollator:
        def __call__(self, features):
            max_length = max(len(feature["input_ids"]) for feature in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                padding = max_length - len(feature["input_ids"])
                batch["input_ids"].append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
                batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
                batch["labels"].append(feature["labels"] + [-100] * padding)
            return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=args.batch,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=25,
        report_to="none",
        eval_strategy="steps",
        eval_steps=args.save_steps,
        per_device_eval_batch_size=args.batch,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=args.save_steps,
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
    # Deliberately do not pass resume_from_checkpoint: v3 starts from v2 weights
    # with fresh optimizer/scheduler state and writes only to the v3 output path.
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "v3_training_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
