# SLM Pretraining on TF1-EN-3M — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pretrain two small Llama-style SLMs (~10M, ~30M) from scratch on a 500k subset of TF1-EN-3M, conditioned on the 5-slot narrative scaffold, then evaluate them scientifically (per ADR-0002) and serve them in the existing app to show a small model rivals Qwen3-4B on fable quality.

**Architecture:** A local, TDD-able data + tokenizer + evaluation pipeline (`scripts/`, `app/`) plus a Colab-T4 training notebook (`notebooks/`). Training reuses the app's prompt format so train==inference; models export to GGUF and run through the existing Ollama-backed app. Evaluation writes `results/eval_summary.json` which the existing Results tab renders (extended for the size/checkpoint ladder).

**Tech Stack:** Python 3.11+, HuggingFace `transformers` (`LlamaForCausalLM`) + `datasets` + `tokenizers`, PyTorch, llama.cpp (GGUF export), Ollama; React+TS+recharts frontend (existing). Tests: pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-08-slm-pretrain-tf1-design.md`. Branch: `feat/slm-pretrain-tf1`.
- Train **from scratch** (no fine-tuning a strong base). Two sizes only: **~10M** and **~30M**.
- Conditioning = the **5 app slots** (character, setting, challenge, outcome, teaching) via **`app.prompt_en.build_fable_prompt`** so training format == app inference format. **Slot dropout** so empty fields work.
- Tokenizer = **custom BPE, vocab ~12k**, trained on the fable corpus, with special tokens for conditioning/story separation.
- Data = **subset ~500k** from the dataset's `train` split (2.8M); held-out eval from its `test` split. Dedup by `prompt_hash`.
- Compute = **Colab free T4**; support checkpoint + resume via Google Drive; save a **step-0 (random-init) checkpoint** as the "before" baseline.
- Recipe = AdamW β(0.9,0.95), wd 0.1, grad-clip 1.0, **WSD** LR schedule, fp16; token budget ~Chinchilla 20 tok/param. **No μP.**
- Evaluation methodology = **ADR-0002 verbatim**: objective metrics (perplexity, Distinct-1/2, Self-BLEU, Flesch Reading Ease) + **LLM-judge panel of 3 distinct-family models** (`qwen3:4b`, `gemma2:2b`, `llama3.2:3b`) scoring the paper's **4 axes** (grammar, creativity, moral_clarity, prompt_adherence), + **weighted Cohen's κ** and **Kendall's τ**; conclude before/after **by rank**, not absolute single-judge scores. Cite Nadas et al. (2025), arXiv:2504.20605.
- App integration: model registry `config/models.json`, new `kind` value `scratch-slm`; SLMs must be selectable and Compare-able against Qwen; `results/eval_summary.json` schema stays backward-compatible (Results tab renders defensively).
- UI copy rules (existing project constraints): English UI, **no emoji** (use react-icons `Md*`), **no em-dash** anywhere in `web/src`.
- TF1 record fields (verified): `prompt` (contains the scaffold), `fable` (story text), `prompt_hash` (dedup). Splits: train 2.8M, validation 100k, test 100k.

---

## File Structure

- `scripts/tf1_pretrain/__init__.py` — package marker.
- `scripts/tf1_pretrain/parse.py` — parse the 5 slots out of a TF1 `prompt` string.
- `scripts/tf1_pretrain/format.py` — build conditional training text (reuse `build_fable_prompt`), slot dropout, length bucketing, loss-mask boundary.
- `scripts/prepare_tf1_pretrain.py` — CLI: stream dataset, dedup, sample, format, write jsonl.
- `scripts/train_tokenizer.py` — CLI: train BPE tokenizer (~12k) on the prepared corpus.
- `notebooks/pretrain_slm_tf1.ipynb` — Colab T4 training notebook (both sizes).
- `app/metrics.py` — reference-free metrics: `distinct_n`, `self_bleu`, `flesch_reading_ease`.
- `app/agreement.py` — `cohen_kappa_weighted`, `kendall_tau`.
- `app/perplexity.py` — `perplexity_from_nll`.
- `scripts/eval_slm.py` — CLI: batch eval → `results/eval_summary.json` (extended schema).
- `web/src/components/ResultsPanel.tsx` — extend for 3-way models + size/checkpoint ladder (existing file).
- Tests under `tests/`.

Dependencies: add optional groups to `pyproject.toml` — `eval = ["textstat>=0.7", "scipy>=1.11"]`. Training deps (`torch`, `transformers`, `datasets`, `tokenizers`) are installed inside Colab and used by the prepare/tokenizer scripts locally only if run locally; prepare/tokenizer tests must not require torch.

---

## Phase P1 — Data & tokenizer

### Task 1: Parse scaffold slots from the TF1 prompt

**Files:**
- Create: `scripts/tf1_pretrain/__init__.py` (empty)
- Create: `scripts/tf1_pretrain/parse.py`
- Test: `tests/test_tf1_parse.py`

**Interfaces:**
- Produces: `parse_slots(prompt: str) -> dict[str, str]` returning keys `character, setting, challenge, outcome, teaching` (missing/blank → `""`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tf1_parse.py
from scripts.tf1_pretrain.parse import parse_slots

SAMPLE = (
    "Create a fable based on the following elements. Weave them naturally into a story:\n"
    "- Main Character: a clever young fox\n"
    "- Setting: a foggy riverside marsh\n"
    "- Challenge: a heron guards the only fish\n"
    "- Outcome: the fox tricks the heron and escapes\n"
    "- Teaching: cleverness beats brute force\n"
    "Formatting requirements: age group B (4-7)...\n"
)

def test_parse_all_slots():
    s = parse_slots(SAMPLE)
    assert s["character"] == "a clever young fox"
    assert s["setting"] == "a foggy riverside marsh"
    assert s["challenge"] == "a heron guards the only fish"
    assert s["outcome"] == "the fox tricks the heron and escapes"
    assert s["teaching"] == "cleverness beats brute force"

def test_parse_missing_slot_is_blank():
    s = parse_slots("- Main Character: a lonely owl\n- Setting: an old oak\n")
    assert s["character"] == "a lonely owl"
    assert s["challenge"] == ""

def test_parse_stops_at_formatting_section():
    # A value must not swallow following "Formatting requirements" text.
    s = parse_slots(SAMPLE)
    assert "Formatting" not in s["teaching"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_tf1_parse.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# scripts/tf1_pretrain/parse.py
"""Extract the 5 narrative scaffold slots from a TF1-EN-3M prompt string."""
import re

_LABELS = {
    "character": "Main Character",
    "setting": "Setting",
    "challenge": "Challenge",
    "outcome": "Outcome",
    "teaching": "Teaching",
}


def parse_slots(prompt: str) -> dict[str, str]:
    """Return the 5 app slots parsed from a TF1 prompt. Missing slot -> ""."""
    out: dict[str, str] = {}
    for key, label in _LABELS.items():
        # Match "- <Label>: <value>" up to end-of-line.
        m = re.search(rf"-\s*{re.escape(label)}\s*:\s*(.+)", prompt)
        out[key] = m.group(1).strip() if m else ""
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_tf1_parse.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/tf1_pretrain/__init__.py scripts/tf1_pretrain/parse.py tests/test_tf1_parse.py
git commit -m "feat: parse TF1 scaffold slots from prompt"
```

---

### Task 2: Conditional formatting + slot dropout + length bucketing

**Files:**
- Create: `scripts/tf1_pretrain/format.py`
- Test: `tests/test_tf1_format.py`

**Interfaces:**
- Consumes: `parse_slots` (Task 1); `app.prompt_en.build_fable_prompt(character, setting, challenge, outcome, teaching, length_hint)` and `app.prompt_en.LENGTH_HINT_EN` (existing — omits empty fields).
- Produces:
  - `length_bucket(word_count: int) -> str` → one of `"short"|"medium"|"long"`.
  - `apply_dropout(slots: dict, rng: random.Random, p: float = 0.3, p_all: float = 0.05) -> dict` → returns a copy with some slots blanked; with prob `p_all` blanks all.
  - `build_training_text(slots: dict, fable: str, length: str) -> tuple[str, int]` → returns `(text, cond_len_chars)` where `text = COND + SEP + fable + END`, and `cond_len_chars` marks where the fable begins (for loss masking). Uses special tokens `SEP="<|story|>"`, `END="<|end|>"`. The conditioning is `build_fable_prompt(**slots, length_hint=LENGTH_HINT_EN[length])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tf1_format.py
import random
from scripts.tf1_pretrain.format import (
    length_bucket, apply_dropout, build_training_text, SEP, END,
)

def test_length_bucket():
    assert length_bucket(100) == "short"
    assert length_bucket(300) == "medium"
    assert length_bucket(700) == "long"

def test_apply_dropout_blanks_some_but_keeps_keys():
    slots = {"character": "a fox", "setting": "a marsh", "challenge": "c",
             "outcome": "o", "teaching": "t"}
    rng = random.Random(0)
    dropped = apply_dropout(slots, rng, p=1.0, p_all=0.0)
    assert set(dropped) == set(slots)           # keys preserved
    assert all(v == "" for v in dropped.values())  # p=1.0 blanks all non-all path
    assert slots["character"] == "a fox"        # original untouched

def test_build_training_text_has_separator_and_fable():
    slots = {"character": "a fox", "setting": "", "challenge": "",
             "outcome": "", "teaching": "be kind"}
    text, cond_len = build_training_text(slots, "Once upon a time. The end.", "short")
    assert SEP in text and text.endswith(END)
    assert text[cond_len:].startswith(SEP)      # fable region begins at the separator
    assert "be kind" in text[:cond_len]         # conditioning contains present slots
    assert "Setting" not in text[:cond_len]     # empty slots omitted by build_fable_prompt
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_tf1_format.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# scripts/tf1_pretrain/format.py
"""Conditional training-text formatting for SLM pretraining."""
import random

from app.prompt_en import build_fable_prompt, LENGTH_HINT_EN

SEP = "<|story|>"
END = "<|end|>"
SLOT_KEYS = ("character", "setting", "challenge", "outcome", "teaching")


def length_bucket(word_count: int) -> str:
    if word_count < 200:
        return "short"
    if word_count < 450:
        return "medium"
    return "long"


def apply_dropout(slots: dict, rng: random.Random, p: float = 0.3,
                  p_all: float = 0.05) -> dict:
    """Return a copy of slots with some values blanked (keys preserved).

    With probability p_all, blank every slot (free-generation example).
    Otherwise blank each slot independently with probability p.
    """
    if rng.random() < p_all:
        return {k: "" for k in slots}
    return {k: ("" if rng.random() < p else v) for k, v in slots.items()}


def build_training_text(slots: dict, fable: str, length: str) -> tuple[str, int]:
    """Return (text, cond_len_chars). text = COND + SEP + fable + END.

    Conditioning reuses the app's build_fable_prompt so training format
    matches inference. cond_len_chars is the index where SEP begins (for
    loss masking the conditioning region).
    """
    cond = build_fable_prompt(
        slots.get("character", ""), slots.get("setting", ""),
        slots.get("challenge", ""), slots.get("outcome", ""),
        slots.get("teaching", ""), LENGTH_HINT_EN[length],
    )
    prefix = cond + "\n"
    text = prefix + SEP + fable.strip() + END
    return text, len(prefix)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_tf1_format.py -v`
Expected: PASS (3 tests).

Note: if `build_fable_prompt`'s positional signature differs, adapt the call; do not change `app/prompt_en.py`.

- [ ] **Step 5: Commit**

```bash
git add scripts/tf1_pretrain/format.py tests/test_tf1_format.py
git commit -m "feat: conditional training-text format with slot dropout"
```

---

### Task 3: Prepare CLI (stream, dedup, sample, write jsonl)

**Files:**
- Create: `scripts/prepare_tf1_pretrain.py`
- Test: `tests/test_prepare_tf1_pretrain.py`

**Interfaces:**
- Consumes: `parse_slots` (Task 1), `apply_dropout`/`build_training_text`/`length_bucket` (Task 2).
- Produces: `build_record(row: dict, rng) -> dict | None` returning `{"text": str, "cond_len": int}` (None if the row lacks a usable fable). A `main(argv)` CLI writing `data/tf1/{train,test}.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_tf1_pretrain.py
import random
from scripts.prepare_tf1_pretrain import build_record

ROW = {
    "prompt": ("- Main Character: a fox\n- Setting: a marsh\n- Challenge: hunger\n"
               "- Outcome: shares food\n- Teaching: sharing is caring\n"),
    "fable": "A fox found food and shared it. Moral: sharing is caring.",
    "prompt_hash": "abc123",
}

def test_build_record_returns_text_and_cond_len():
    rec = build_record(ROW, random.Random(0))
    assert rec is not None
    assert "text" in rec and "cond_len" in rec
    assert rec["text"][rec["cond_len"]:].startswith("<|story|>")

def test_build_record_skips_empty_fable():
    row = {**ROW, "fable": "   "}
    assert build_record(row, random.Random(0)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_prepare_tf1_pretrain.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# scripts/prepare_tf1_pretrain.py
"""Prepare a conditional pretraining corpus from klusai/ds-tf1-en-3m.

Streams the dataset, parses the 5 scaffold slots, applies slot dropout,
formats train text with the app's prompt format, dedups by prompt_hash,
and writes JSONL. Training deps (datasets) only needed when actually
streaming; build_record itself is pure and unit-tested.
"""
import argparse
import json
import random
from pathlib import Path

from scripts.tf1_pretrain.parse import parse_slots
from scripts.tf1_pretrain.format import (
    apply_dropout, build_training_text, length_bucket,
)


def build_record(row: dict, rng: random.Random) -> dict | None:
    fable = (row.get("fable") or "").strip()
    if not fable:
        return None
    slots = parse_slots(row.get("prompt") or "")
    slots = apply_dropout(slots, rng)
    length = length_bucket(len(fable.split()))
    text, cond_len = build_training_text(slots, fable, length)
    return {"text": text, "cond_len": cond_len}


def _write_split(rows, out_path: Path, n: int, seed: int) -> int:
    rng = random.Random(seed)
    seen: set[str] = set()
    written = 0
    with out_path.open("w") as f:
        for row in rows:
            h = row.get("prompt_hash")
            if h in seen:
                continue
            seen.add(h)
            rec = build_record(row, rng)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= n:
                break
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=500_000)
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--out", default="data/tf1")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    from datasets import load_dataset  # imported lazily (training dep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = load_dataset("klusai/ds-tf1-en-3m", split="train", streaming=True)
    n_tr = _write_split(train, out / "train.jsonl", args.train_n, args.seed)
    test = load_dataset("klusai/ds-tf1-en-3m", split="test", streaming=True)
    n_te = _write_split(test, out / "test.jsonl", args.test_n, args.seed + 1)
    print(f"wrote train={n_tr} test={n_te} to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_prepare_tf1_pretrain.py -v`
Expected: PASS (2 tests). (The `datasets` import is lazy, so tests pass without it.)

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_tf1_pretrain.py tests/test_prepare_tf1_pretrain.py
git commit -m "feat: TF1 pretraining corpus prepare CLI"
```

---

### Task 4: Train the BPE tokenizer

**Files:**
- Create: `scripts/train_tokenizer.py`
- Test: `tests/test_train_tokenizer.py`

**Interfaces:**
- Produces: `train_bpe(texts: list[str], vocab_size: int, special_tokens: list[str]) -> Tokenizer` (HF `tokenizers.Tokenizer`) and a `main(argv)` CLI reading `data/tf1/train.jsonl` and saving `data/tf1/tokenizer.json`.

- [ ] **Step 1: Write the failing test** (skips cleanly if `tokenizers` is absent)

```python
# tests/test_train_tokenizer.py
import pytest

tokenizers = pytest.importorskip("tokenizers")
from scripts.train_tokenizer import train_bpe
from scripts.tf1_pretrain.format import SEP, END


def test_train_bpe_roundtrip_and_specials():
    texts = ["a fox shared food. moral: be kind."] * 50
    tok = train_bpe(texts, vocab_size=300, special_tokens=[SEP, END, "<|pad|>"])
    assert tok.token_to_id(SEP) is not None
    ids = tok.encode(f"a fox {SEP} be kind {END}").ids
    assert len(ids) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_train_tokenizer.py -v`
Expected: FAIL if `tokenizers` installed (function missing); else SKIP.

- [ ] **Step 3: Implement**

```python
# scripts/train_tokenizer.py
"""Train a small BPE tokenizer on the fable corpus."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from scripts.tf1_pretrain.format import SEP, END

PAD = "<|pad|>"


def train_bpe(texts, vocab_size: int, special_tokens):
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|unk|>", PAD, *special_tokens],
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tf1/train.jsonl")
    ap.add_argument("--out", default="data/tf1/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=12000)
    args = ap.parse_args(argv)

    def it():
        with open(args.data) as f:
            for line in f:
                yield json.loads(line)["text"]

    tok = train_bpe(it(), args.vocab_size, [SEP, END])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer ({args.vocab_size} vocab) to {args.out}")


if __name__ == "__main__":
    main()
```

Note: `train_from_iterator` accepts any iterable of strings, so `main` streams from the jsonl; the test passes a list.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_train_tokenizer.py -v`
Expected: PASS (or SKIP without `tokenizers`).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_tokenizer.py tests/test_train_tokenizer.py
git commit -m "feat: train small BPE tokenizer for fable corpus"
```

---

## Phase P2 — Training notebook (Colab T4)

### Task 5: Pretraining notebook

**Files:**
- Create: `notebooks/pretrain_slm_tf1.ipynb`

**Interfaces:**
- Consumes: `data/tf1/{train,test}.jsonl` (Task 3), `data/tf1/tokenizer.json` (Task 4).
- Produces: per size, an HF model dir + checkpoints (incl. step-0), a `loss_log.json` (list of `{step, train_loss, val_loss}`), and a GGUF file + Ollama `Modelfile`.

This task delivers a notebook, not unit-tested code. It MUST contain these cells with runnable code; verification is a **tiny smoke run** (below). Each numbered item is one cell.

- [ ] **Step 1: Cell — install & imports**

```python
%pip -q install "transformers>=4.44" "datasets>=2.20" "tokenizers>=0.19" torch accelerate
import json, math, random, os
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          Trainer, TrainingArguments)
```

- [ ] **Step 2: Cell — HYPERPARAMS (edit here)**

```python
SIZE = "10M"   # "10M" or "30M"
CONFIGS = {
  "10M": dict(hidden_size=320, intermediate_size=1280, num_hidden_layers=6,
              num_attention_heads=8, num_key_value_heads=2),
  "30M": dict(hidden_size=512, intermediate_size=2048, num_hidden_layers=8,
              num_attention_heads=8, num_key_value_heads=2),
}
SEQ_LEN = 512
TOKEN_BUDGET = {"10M": 200_000_000, "30M": 600_000_000}[SIZE]
PEAK_LR = 3e-3
WARMUP_FRAC, DECAY_FRAC = 0.02, 0.20
GLOBAL_BATCH_TOKENS = 262_144          # ~0.26M tokens/step
DRIVE_DIR = "/content/drive/MyDrive/slm_tf1"
SMOKE = False                          # True = tiny run to validate pipeline
```

- [ ] **Step 3: Cell — mount Drive + load tokenizer**

```python
from google.colab import drive; drive.mount("/content/drive")
tok = PreTrainedTokenizerFast(tokenizer_file="data/tf1/tokenizer.json",
        unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")
VOCAB = tok.vocab_size
```

- [ ] **Step 4: Cell — dataset with loss masking**

Tokenize `text`; build `labels` = input_ids but with the conditioning region (before `cond_len` mapped to token index) set to `-100` so loss is computed only on the fable. Pack/truncate to `SEQ_LEN`.

```python
def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f]
train_rows = load_jsonl("data/tf1/train.jsonl")
if SMOKE: train_rows = train_rows[:2000]

def encode(row):
    ids = tok(row["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
    # token index of the story separator = mask everything up to & incl. cond
    cond_ids = tok(row["text"][:row["cond_len"]])["input_ids"]
    k = min(len(cond_ids), len(ids))
    labels = [-100]*k + ids[k:]
    return {"input_ids": ids, "labels": labels}
train_ds = [encode(r) for r in train_rows]
```

Use a data collator that pads `input_ids` with `pad` and `labels` with `-100`.

- [ ] **Step 5: Cell — model from scratch + WSD schedule**

```python
cfg = LlamaConfig(vocab_size=VOCAB, max_position_embeddings=SEQ_LEN,
                  tie_word_embeddings=True, **CONFIGS[SIZE])
model = LlamaForCausalLM(cfg)
print("params(M):", sum(p.numel() for p in model.parameters())/1e6)

steps = max(1, TOKEN_BUDGET // GLOBAL_BATCH_TOKENS) if not SMOKE else 50
def wsd(step):
    w, d = int(WARMUP_FRAC*steps), int(DECAY_FRAC*steps)
    if step < w: return step/max(1,w)
    if step > steps-d: return max(0.0, (steps-step)/max(1,d))
    return 1.0
```

- [ ] **Step 6: Cell — save step-0 checkpoint (the "before" baseline)**

```python
os.makedirs(f"{DRIVE_DIR}/{SIZE}/step-0", exist_ok=True)
model.save_pretrained(f"{DRIVE_DIR}/{SIZE}/step-0"); tok.save_pretrained(f"{DRIVE_DIR}/{SIZE}/step-0")
```

- [ ] **Step 7: Cell — Trainer (fp16, checkpoints, resume, loss log)**

```python
args = TrainingArguments(
    output_dir=f"{DRIVE_DIR}/{SIZE}", max_steps=steps, fp16=True,
    per_device_train_batch_size=16, gradient_accumulation_steps=32,
    learning_rate=PEAK_LR, weight_decay=0.1, max_grad_norm=1.0,
    lr_scheduler_type="constant", warmup_steps=0,     # WSD applied via callback
    logging_steps=25, save_steps=max(1, steps//5), save_total_limit=6,
    report_to=[])
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
# Attach a LambdaLR(wsd) via trainer.create_optimizer_and_scheduler override or a callback.
trainer.train(resume_from_checkpoint=bool(os.listdir(f"{DRIVE_DIR}/{SIZE}")) and not SMOKE)
json.dump(trainer.state.log_history, open(f"{DRIVE_DIR}/{SIZE}/loss_log.json","w"))
```

- [ ] **Step 8: Cell — export GGUF + Ollama Modelfile**

```python
!git clone -q https://github.com/ggerganov/llama.cpp
!python llama.cpp/convert_hf_to_gguf.py {DRIVE_DIR}/{SIZE} --outfile {DRIVE_DIR}/slm-{SIZE}.gguf --outtype q8_0
open(f"{DRIVE_DIR}/Modelfile-{SIZE}","w").write(
  f"FROM ./slm-{SIZE}.gguf\nPARAMETER temperature 0.8\nPARAMETER top_p 0.9\n"
  f"PARAMETER repeat_penalty 1.3\nPARAMETER stop \"<|end|>\"\nPARAMETER num_ctx 512\n")
```

- [ ] **Step 9: Verify — smoke run**

Set `SMOKE = True`, run all cells for `SIZE="10M"`. Expected: model builds, `params(M)` prints ~10, training runs 50 steps with **decreasing** train loss, checkpoints + `loss_log.json` written, GGUF export produces a file. This validates the pipeline before the full multi-hour run.

- [ ] **Step 10: Commit the notebook** (committed with `SMOKE=True` and outputs cleared)

```bash
git add notebooks/pretrain_slm_tf1.ipynb
git commit -m "feat: Colab T4 SLM pretraining notebook (10M/30M, WSD, GGUF export)"
```

- [ ] **Step 11: Full runs (interactive, tracked in the ledger, not a code change)**

Run `SIZE="10M"` then `SIZE="30M"` with `SMOKE=False` on Colab T4. Confirm loss decreases and sample generations become coherent fables. Download GGUFs. (Executed during Phase P4 integration.)

---

## Phase P3 — Scientific evaluation

### Task 6: Reference-free metrics

**Files:**
- Modify: `pyproject.toml` (add `eval` optional deps)
- Create: `app/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `distinct_n(texts: list[str], n: int) -> float` — ratio of unique n-grams to total n-grams (0..1).
  - `self_bleu(texts: list[str], n: int = 4) -> float` — mean pairwise BLEU-n (0..1); higher = more repetitive.
  - `flesch_reading_ease(text: str) -> float` — via `textstat`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from app.metrics import distinct_n, self_bleu, flesch_reading_ease

def test_distinct_n_all_unique():
    assert distinct_n(["a b c d"], 1) == 1.0

def test_distinct_n_with_repeats():
    # tokens: a a a a -> 1 unique / 4 total = 0.25
    assert distinct_n(["a a a a"], 1) == 0.25

def test_self_bleu_identical_texts_high():
    v = self_bleu(["the fox ran fast", "the fox ran fast"], n=2)
    assert v > 0.9

def test_self_bleu_disjoint_texts_low():
    v = self_bleu(["alpha beta gamma delta", "one two three four"], n=2)
    assert v < 0.1

def test_flesch_returns_number():
    assert isinstance(flesch_reading_ease("The fox ran. The end."), float)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Add deps + implement**

Add to `pyproject.toml` under `[project.optional-dependencies]`: `eval = ["textstat>=0.7", "scipy>=1.11"]`. Install: `pip install -e ".[eval]"`.

```python
# app/metrics.py
"""Reference-free diversity + readability metrics (ADR-0002)."""
from collections import Counter
from itertools import combinations

import textstat


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        grams = _ngrams(t.split(), n)
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def _bleu_n(cand: list[str], ref: list[str], n: int) -> float:
    if len(cand) < n:
        return 0.0
    cg, rg = Counter(_ngrams(cand, n)), Counter(_ngrams(ref, n))
    overlap = sum((cg & rg).values())
    return overlap / max(1, sum(cg.values()))


def self_bleu(texts: list[str], n: int = 4) -> float:
    toks = [t.split() for t in texts]
    pairs = list(combinations(range(len(toks)), 2))
    if not pairs:
        return 0.0
    return sum(_bleu_n(toks[i], toks[j], n) for i, j in pairs) / len(pairs)


def flesch_reading_ease(text: str) -> float:
    return float(textstat.flesch_reading_ease(text))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml app/metrics.py tests/test_metrics.py
git commit -m "feat: reference-free diversity/readability metrics"
```

---

### Task 7: Inter-judge agreement

**Files:**
- Create: `app/agreement.py`
- Test: `tests/test_agreement.py`

**Interfaces:**
- Produces:
  - `cohen_kappa_weighted(a: list[int], b: list[int], max_score: int = 10) -> float` — quadratic-weighted κ.
  - `kendall_tau(a: list[float], b: list[float]) -> float` — rank correlation (via `scipy.stats.kendalltau`, NaN → 0.0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agreement.py
from app.agreement import cohen_kappa_weighted, kendall_tau

def test_kappa_perfect_agreement():
    assert cohen_kappa_weighted([1,5,9], [1,5,9]) == 1.0

def test_kappa_close_scores_positive():
    assert cohen_kappa_weighted([8,9,7,10], [7,9,8,10]) > 0.0

def test_kendall_tau_monotonic():
    assert kendall_tau([1,2,3,4], [1,2,3,4]) == 1.0

def test_kendall_tau_reversed():
    assert kendall_tau([1,2,3,4], [4,3,2,1]) == -1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_agreement.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# app/agreement.py
"""Inter-judge agreement metrics (ADR-0002): weighted Cohen's kappa + Kendall's tau."""
import math

from scipy.stats import kendalltau


def cohen_kappa_weighted(a: list[int], b: list[int], max_score: int = 10) -> float:
    """Quadratic-weighted Cohen's kappa over integer scores in [0, max_score]."""
    k = max_score + 1
    n = len(a)
    if n == 0:
        return 0.0
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1.0
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * row[i] * col[j] / n for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1.0 - num / den


def kendall_tau(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    tau, _ = kendalltau(a, b)
    return 0.0 if (tau is None or math.isnan(tau)) else float(tau)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_agreement.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/agreement.py tests/test_agreement.py
git commit -m "feat: weighted Cohen kappa + Kendall tau agreement metrics"
```

---

### Task 8: Perplexity helper

**Files:**
- Create: `app/perplexity.py`
- Test: `tests/test_perplexity.py`

**Interfaces:**
- Produces: `perplexity_from_nll(total_nll: float, total_tokens: int) -> float` = `exp(total_nll/total_tokens)`; `aggregate_nll(per_seq: list[tuple[float,int]]) -> float` summing token-weighted.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perplexity.py
import math
from app.perplexity import perplexity_from_nll, aggregate_nll

def test_perplexity_uniform():
    # nll per token = ln(2) -> perplexity = 2
    assert abs(perplexity_from_nll(math.log(2)*10, 10) - 2.0) < 1e-6

def test_perplexity_zero_tokens_safe():
    assert perplexity_from_nll(0.0, 0) == float("inf")

def test_aggregate_nll_token_weighted():
    assert aggregate_nll([(2.0, 2), (3.0, 3)]) == 5.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_perplexity.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# app/perplexity.py
"""Held-out perplexity aggregation (mockable; model forward done by caller)."""
import math


def aggregate_nll(per_seq: list[tuple[float, int]]) -> float:
    return sum(nll for nll, _ in per_seq)


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_perplexity.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/perplexity.py tests/test_perplexity.py
git commit -m "feat: perplexity aggregation helper"
```

---

### Task 9: Batch eval CLI → eval_summary.json

**Files:**
- Create: `scripts/eval_slm.py`
- Test: `tests/test_eval_slm.py`

**Interfaces:**
- Consumes: `app.judge.evaluate(story, prompt, model, gen)` (existing → `{grammar,creativity,moral_clarity,prompt_adherence,overall,rationale}`); `app.metrics.*`, `app.agreement.*`; `app.ollama_client.generate` for generation.
- Produces:
  - `aggregate_axis_scores(panel: dict[str, list[dict]]) -> dict` → mean per axis across judges per model.
  - `panel_agreement(panel: dict[str, list[dict]]) -> dict` → `{cohen_kappa, kendall_tau}` averaged over judge pairs on `overall`.
  - `conclude_by_rank(model_scores: dict[str, float]) -> dict` → `{winner, by_rank, notes}`.
  - `build_summary(...) -> dict` producing the extended `eval_summary.json` (below). `main(argv)` runs generation + judging (integration).

**Extended `eval_summary.json` schema** (backward-compatible superset of what ResultsPanel already reads):

```jsonc
{
  "models": ["slm-10m", "slm-30m", "qwen3-4b"],   // NEW: N-way
  "objective": { "slm-10m": {perplexity, distinct_1, distinct_2, self_bleu, flesch_reading_ease}, ... },
  "judge_panel": { "judges": [...], "slm-10m": {grammar,creativity,moral_clarity,prompt_adherence,overall}, ... },
  "agreement": { "cohen_kappa": .., "kendall_tau": .. },
  "conclusion": { "winner": "..", "by_rank": "..", "notes": ".." },
  "size_ladder": [ {"model": "slm-10m", "params_m": 10, "overall": ..}, {"model": "slm-30m", "params_m": 30, "overall": ..} ],  // NEW
  "checkpoint_curve": { "slm-30m": [ {"step": 0, "overall": ..}, ... ] },   // NEW (from step-0 + checkpoints)
  "loss_curve": [ {"step": .., "loss": ..} ]        // optional, from loss_log.json
}
```

- [ ] **Step 1: Write the failing test** (pure aggregation only; generation/judging are integration)

```python
# tests/test_eval_slm.py
from scripts.eval_slm import aggregate_axis_scores, conclude_by_rank

def test_aggregate_axis_scores_means_over_judges():
    panel = {
      "j1": [{"grammar":8,"creativity":6,"moral_clarity":10,"prompt_adherence":9,"overall":8.25}],
      "j2": [{"grammar":10,"creativity":8,"moral_clarity":10,"prompt_adherence":9,"overall":9.25}],
    }
    agg = aggregate_axis_scores(panel)
    assert agg["grammar"] == 9.0 and agg["overall"] == 8.75

def test_conclude_by_rank_picks_highest():
    c = conclude_by_rank({"slm-30m": 8.9, "qwen3-4b": 9.0, "slm-10m": 8.1})
    assert c["winner"] == "qwen3-4b"
    assert "rank" in c["by_rank"].lower() or "qwen3-4b" in c["by_rank"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_eval_slm.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (aggregation pure; generation/judging behind `main`)

```python
# scripts/eval_slm.py
"""Batch scientific evaluation (ADR-0002): SLMs vs Qwen on held-out TF1 test.

Cite: Nadas et al. (2025), TF1-EN-3M, arXiv:2504.20605.
"""
import argparse
import json
from pathlib import Path

from app import judge, ollama_client
from app.metrics import distinct_n, self_bleu, flesch_reading_ease
from app.agreement import cohen_kappa_weighted, kendall_tau

AXES = ["grammar", "creativity", "moral_clarity", "prompt_adherence"]


def aggregate_axis_scores(panel: dict) -> dict:
    rows = [r for judge_rows in panel.values() for r in judge_rows]
    out = {a: round(sum(r[a] for r in rows) / len(rows), 3) for a in AXES}
    out["overall"] = round(sum(out[a] for a in AXES) / len(AXES), 3)
    return out


def panel_agreement(overall_by_judge: dict) -> dict:
    judges = list(overall_by_judge)
    kappas, taus = [], []
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            a = [int(round(x)) for x in overall_by_judge[judges[i]]]
            b = [int(round(x)) for x in overall_by_judge[judges[j]]]
            kappas.append(cohen_kappa_weighted(a, b))
            taus.append(kendall_tau(overall_by_judge[judges[i]], overall_by_judge[judges[j]]))
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    return {"cohen_kappa": avg(kappas), "kendall_tau": avg(taus)}


def conclude_by_rank(model_overall: dict) -> dict:
    ranked = sorted(model_overall.items(), key=lambda kv: kv[1], reverse=True)
    winner = ranked[0][0]
    order = " > ".join(f"{m} ({s:.2f})" for m, s in ranked)
    return {"winner": winner,
            "by_rank": f"By rank: {order}",
            "notes": "Conclusion by rank across judges, not absolute single-judge scores."}
# main(): load data/tf1/test.jsonl prompts; for each model generate via ollama_client;
# for each judge model call judge.evaluate; compute objective + panel + agreement +
# size_ladder + checkpoint_curve; write results/eval_summary.json. (Integration.)
```

The full `main` (generation + judging loops, writing the summary) is integration code; implement it following the interfaces above and the schema. Run it in Phase P4 after models exist.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_eval_slm.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_slm.py tests/test_eval_slm.py
git commit -m "feat: batch eval aggregation for SLM vs Qwen (ADR-0002)"
```

---

## Phase P4 — App integration

### Task 10: Register SLMs + run the real pipeline

**Files:**
- Modify: `config/models.json`
- Test: `tests/test_api_en.py` (extend)

This task registers the trained models and runs the real train→export→eval pipeline (interactive). Registry edits are code; the runs are tracked in the ledger.

- [ ] **Step 1: Write the failing test** (registry exposes the SLMs with the new kind)

```python
# add to tests/test_api_en.py
def test_models_includes_scratch_slms():
    r = client.get("/models")
    ids = {m["id"] for m in r.json()}
    assert {"slm-10m", "slm-30m"}.issubset(ids)
    kinds = {m["id"]: m["kind"] for m in r.json()}
    assert kinds["slm-10m"] == "scratch-slm"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_api_en.py::test_models_includes_scratch_slms -v`
Expected: FAIL (ids absent).

- [ ] **Step 3: Add registry entries**

Append to `config/models.json` (after `ollama create slm-10m/slm-30m` from the notebook Modelfiles):

```json
{"id": "slm-10m", "name": "Fable-SLM 10M (from scratch)", "ollama": "slm-10m", "kind": "scratch-slm", "desc": "10M-param Llama-style SLM pretrained from scratch on TF1"},
{"id": "slm-30m", "name": "Fable-SLM 30M (from scratch)", "ollama": "slm-30m", "kind": "scratch-slm", "desc": "30M-param Llama-style SLM pretrained from scratch on TF1"}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_api_en.py::test_models_includes_scratch_slms -v`
Expected: PASS.

- [ ] **Step 5: Commit + run pipeline**

```bash
git add config/models.json tests/test_api_en.py
git commit -m "feat: register from-scratch SLMs in model registry"
```

Then interactively (tracked in ledger): run the notebook full trainings (Task 5 Step 11), `ollama create` both GGUFs, run `python3 -m scripts.prepare_tf1_pretrain` (once), and `python3 -m scripts.eval_slm` to produce `results/eval_summary.json`.

---

### Task 11: Results tab — size + checkpoint ladder

**Files:**
- Modify: `web/src/components/ResultsPanel.tsx`
- (Reuse existing `web/src/components/EvalRadar.tsx`.)

**Interfaces:**
- Consumes: extended `eval_summary.json` (Task 9) via existing `fetchResults()`.
- Produces: rendering for `models` (N-way objective + judge table), `size_ladder` (recharts line: overall vs params_m), and `checkpoint_curve` (recharts line: overall vs step) — all rendered defensively (only if the section is present), matching the existing empty-state/placeholder behavior.

- [ ] **Step 1: Extend the objective/judge tables to N models**

Render one column per entry in `data.models` (fallback to `["base","finetuned"]` for old files). Keep the existing κ/τ + conclusion blocks unchanged.

- [ ] **Step 2: Add the size-ladder chart**

If `data.size_ladder` present, render a recharts `LineChart` (x = `params_m`, y = `overall`) titled "Quality vs model size". Wrap in an `overflow-x: auto` container.

- [ ] **Step 3: Add the checkpoint-curve chart**

If `data.checkpoint_curve` present, render a `LineChart` per model (x = `step`, y = `overall`) titled "Quality vs training step (before -> after)".

- [ ] **Step 4: Build smoke**

```bash
export PATH="/Users/trieulh/.nvm/versions/node/v25.7.0/bin:$PATH"
cd web && npm run build
```
Expected: tsc strict clean + vite build succeeds. No emoji, no em-dash added (English UI, react-icons only).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/ResultsPanel.tsx
git commit -m "feat: Results tab renders SLM size + checkpoint ladders"
```

---

## Self-Review

**Spec coverage:**
- §3 architecture (10M/30M Llama) → Task 5. ✅
- §4 data pipeline (parse, dropout, subset, dedup, tokenizer, splits) → Tasks 1-4. ✅
- §5 training recipe (WSD, mask-loss, checkpoints incl step-0, GGUF export, resume, no μP) → Task 5. ✅
- §6 evaluation (perplexity, Distinct/Self-BLEU/Flesch, 3-judge panel, κ/τ, rank conclusion) → Tasks 6-9. ✅
- §7 integration (registry `scratch-slm`, Compare vs Qwen, Results ladder) → Tasks 10-11. ✅
- §8 phases P1-P4 → Tasks map 1-4 / 5 / 6-9 / 10-11. ✅

**Placeholder scan:** The only non-code deliverables are the notebook (Task 5, cells given + smoke verification) and the eval `main` generation loop (Task 9, interfaces + schema given, integration). These are explicitly integration/interactive, not vague placeholders. All unit-testable functions have complete code + tests.

**Type consistency:** `parse_slots` keys (character/setting/challenge/outcome/teaching) are consumed unchanged by `build_training_text` and `build_record`. `SEP`/`END` defined in `format.py` and reused in tokenizer + notebook. Judge axis names (`grammar/creativity/moral_clarity/prompt_adherence/overall`) match `app.judge` and `aggregate_axis_scores`. `eval_summary.json` keys match ResultsPanel consumption (existing keys preserved; new keys `models`/`size_ladder`/`checkpoint_curve` added defensively).

---

## Notes for the executor

- Tasks 1-4, 6-9 are TDD and run locally (no torch needed for tests; `tokenizers`/`textstat`/`scipy` gated or in the `eval` extra).
- Task 5 (notebook) and the real runs in Task 10 require Colab T4 + Ollama and are interactive — do the smoke run before the multi-hour full runs.
- Keep `data/` and `*.gguf` git-ignored (already in `.gitignore`).
