"""Generate matched v2/v3 stories from TF1 rows unseen during v2 training."""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_tf1 import TF1_DATASET, _story_of, parse_elements, rec_is_valid


def load_unseen_controls(skip_valid: int, count: int) -> list[dict]:
    from datasets import load_dataset

    controls = []
    valid_seen = 0
    dataset = load_dataset(TF1_DATASET, split="train", streaming=True)
    for row in dataset:
        if not rec_is_valid(row):
            continue
        if valid_seen < skip_valid:
            valid_seen += 1
            if valid_seen % 25_000 == 0:
                print(f"scanned {valid_seen}/{skip_valid} prior valid rows", flush=True)
            continue
        character, moral = parse_elements(row)
        controls.append(
            {
                "character": character,
                "moral": moral,
                "prompt": (
                    f"<char> {character} </char>\n"
                    f"<moral> {moral} </moral>\n"
                    "<story>\n"
                ),
                "reference_story": _story_of(row),
                "source": f"{TF1_DATASET}:valid-index-{valid_seen}",
            }
        )
        valid_seen += 1
        if len(controls) == count:
            return controls
    raise RuntimeError(f"Only found {len(controls)} controls after valid index {skip_valid}")


def generate_for_model(label, model_path, controls, args):
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = "</story>"
    tokenizer.eos_token = "</story>"
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda").eval()
    rows = []

    for start in range(0, len(controls), args.batch):
        batch_controls = controls[start : start + args.batch]
        torch.manual_seed(args.seed + start)
        torch.cuda.manual_seed_all(args.seed + start)
        encoded = tokenizer(
            [row["prompt"] for row in batch_controls],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[:, encoded["input_ids"].shape[1] :]
        for control, tokens in zip(batch_controls, generated):
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            ended = "</story>" in text
            story = text.split("</story>", 1)[0].strip()
            rows.append(
                {
                    **control,
                    "model": label,
                    "seed": args.seed + start,
                    "output_tokens": int(tokens.shape[0]),
                    "ended": ended,
                    "story": story,
                }
            )
        print(f"{label}: {min(start + args.batch, len(controls))}/{len(controls)}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--skip-valid", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    controls = load_unseen_controls(args.skip_valid, args.controls)
    rows = []
    rows.extend(generate_for_model("v2", args.v2, controls, args))
    rows.extend(generate_for_model("v3-full", args.v3, controls, args))
    result = {
        "kind": "v3-full-generation-comparison",
        "controls": {
            "dataset": TF1_DATASET,
            "skipped_valid_rows": args.skip_valid,
            "count": len(controls),
        },
        "settings": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
            "max_new_tokens": args.max_new_tokens,
            "batch": args.batch,
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
