"""Colab training + generation for the 200M fable model (single-script run).

Streams a subset of TF1-EN-3M, trains a ~200M GPT2-style LM on (character, moral)
seeds, generates example fables, and writes metrics to <out>/eval_summary.json.

Run on Colab via:
  colab upload -s trainer scripts/train_local.py /content/scripts/train_local.py
  colab exec -s trainer -f scripts/train_local.py --timeout 5400
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/content")  # Colab: scripts uploaded to /content/scripts
import torch
from tokenizers import Tokenizer
from transformers import (
    PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline,
)
from datasets import Dataset

from scripts.prepare_tf1 import iter_tf1, prepare_bpe
from scripts.metrics import aggregate_metrics


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fables", type=int, default=200_000)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--n-embd", type=int, default=1024)
    ap.add_argument("--n-layer", type=int, default=16)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/content/fable200m")
    args = ap.parse_args()

    dev = pick_device()
    print(f"[device] {dev}")

    # load HF token if present
    if os.path.exists("/content/hf_token.txt"):
        os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()
        print("HF_TOKEN loaded")

    out = args.out
    os.makedirs(out, exist_ok=True)
    fables_jsonl = os.path.join(out, "fables.jsonl")

    if not os.path.exists(fables_jsonl):
        print(f"[1/4] pulling {args.n_fables} fables from TF1-EN-3M ...")
        recs = list(iter_tf1(args.n_fables, seed=args.seed))
        print(f"      got {len(recs)} valid records")
        prepare_bpe(recs, out, vocab_size=args.vocab_size)
    else:
        print(f"[1/4] reusing cached {fables_jsonl}")

    print("[2/4] loading tokenizer + building dataset ...")
    tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf_tok.pad_token = "</story>"
    hf_tok.eos_token = "</story>"

    texts = [json.loads(l) for l in open(fables_jsonl) if l.strip()]
    ds = Dataset.from_dict({"text": texts})
    def encode(ex):
        ids = tok.encode(ex["text"]).ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}
    ds = ds.map(encode, remove_columns=["text"]).train_test_split(test_size=0.05)

    cfg = GPT2Config(
        vocab_size=tok.get_vocab_size(), n_positions=args.block_size,
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=tok.get_vocab()["<story>"],
        eos_token_id=tok.get_vocab()["</story>"],
    )
    model = GPT2LMHeadModel(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      model params: {n_params/1e6:.1f}M")

    print("[3/4] training ...")
    collator = DataCollatorForLanguageModeling(tokenizer=hf_tok, mlm=False)
    targs = TrainingArguments(
        output_dir=os.path.join(out, "ckpt"), per_device_train_batch_size=args.batch,
        max_steps=args.max_steps, learning_rate=args.lr,
        warmup_steps=max(10, args.max_steps // 10), lr_scheduler_type="cosine",
        logging_steps=50, report_to="none", save_strategy="steps", save_steps=500,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["test"], data_collator=collator)
    # Resume from the latest checkpoint in output_dir if one exists (enables
    # resuming after a VM reclaim when output_dir is on mounted Drive). On a
    # fresh start (no checkpoint yet) begin from scratch.
    import glob as _glob, re as _re
    _ckpts = sorted(_glob.glob(os.path.join(out, "ckpt", "checkpoint-*")),
                    key=lambda p: int(_re.findall(r"\d+", p)[-1]))
    _resume = _ckpts[-1] if _ckpts else False
    if _resume:
        print(f"[3/4] resuming from {_resume}")
    trainer.train(resume_from_checkpoint=_resume)
    trainer.save_model(os.path.join(out, "ckpt"))
    hf_tok.save_pretrained(os.path.join(out, "ckpt"))

    print("[4/4] generating example fables ...")
    pipe = pipeline("text-generation", model=model, tokenizer=hf_tok, device=dev)
    PREFIX = "<char> {character} </char>\n<moral> {moral} </moral>\n<story>\n"
    SEEDS = [
        {"character": "a clever fox", "moral": "cleverness beats brute force"},
        {"character": "a brave little mouse", "moral": "kindness returns to those who give it"},
        {"character": "a proud lion", "moral": "pride comes before a fall"},
        {"character": "an honest ant", "moral": "hard work pays off"},
    ]
    stories = []
    for s in SEEDS:
        prompt = PREFIX.format(**s)
        out_text = pipe(prompt, max_new_tokens=150, do_sample=True,
                        temperature=0.8, top_p=0.9, repetition_penalty=1.3)[0]["generated_text"]
        story = out_text.split("<story>", 1)[-1].split("</story>")[0].strip()
        stories.append(story)
        print(f"\n### {s['character']} | {s['moral']}\n{story}\n")

    metrics = aggregate_metrics(stories)
    summary = {"model": f"fable-{int(n_params//1_000_000)}M",
               "n_fables": len(texts), "params_M": round(n_params / 1e6, 1),
               "max_steps": args.max_steps, "device": dev, "metrics": metrics,
               "examples": [{"seed": s, "story": st} for s, st in zip(SEEDS, stories)]}
    with open(os.path.join(out, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("metrics:", json.dumps(metrics))
    print(f"wrote {os.path.join(out, 'eval_summary.json')}")


if __name__ == "__main__":
    main()
