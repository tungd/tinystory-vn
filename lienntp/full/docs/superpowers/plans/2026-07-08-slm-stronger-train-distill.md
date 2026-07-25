# Stronger SLM Training (full-Chinchilla 30M) + Distillation to 10M — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the v1 undertrain by training the 30M SLM to a full-Chinchilla token budget, then distill it into a 10M student, so the from-scratch SLMs move closer to Qwen on fable quality.

**Architecture:** Reuse the v1 pipeline (`scripts/tf1_pretrain/*`, tokenizer, `scripts/eval_slm.py`, `notebooks/pretrain_slm_tf1.ipynb`). Add: a quality filter + larger subset to the data prep (local, TDD); a full-Chinchilla training config with Drive checkpoint/resume (notebook, interactive on Colab T4); a token-level knowledge-distillation loss (local, TDD) wired into a notebook distillation cell (interactive); then batch eval + registry + Results.

**Tech Stack:** Python 3.11+, HuggingFace `transformers`/`datasets`/`tokenizers`, PyTorch, llama.cpp (GGUF), Ollama; pytest. Training runs on Colab T4.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-08-slm-stronger-train-distill-design.md`. Branch: `feat/slm-pretrain-tf1`. This is **v2** — reuse v1 code, do not rewrite the pipeline/tokenizer/arch.
- Teacher = **30M** from scratch to **~600M tokens** (Chinchilla 20x); data = **~600k unique** subset, **≤4 epochs** (Muennighoff); loss-mask the conditioning prefix.
- Student = **10M** via **token-level KD** from the 30M teacher: `L = alpha·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T)) + (1-alpha)·CE(z_s, y)`, **T=2.0, alpha=0.5**, distilled only on story tokens (conditioning masked).
- Recipe: AdamW β=(0.9,0.95), wd 0.1, grad-clip 1.0, **WSD** schedule, fp16, **peak LR 2e-3** (v2), seq 512, vocab ~12k.
- Compute: **Colab free T4** + **checkpoint/resume to Google Drive** (`/content/drive/MyDrive/slm_tf1`).
- Two GGUF export fixes are mandatory (already in the notebook): transformers-5.x `tokenizer_class="PreTrainedTokenizerFast"` rewrite; llama.cpp custom-BPE pre-tokenizer patch (chkhsh -> "gpt-2"). Modelfile `TEMPLATE """{{ .Prompt }}\n<|story|>"""` (no chat wrapper).
- Evaluation = ADR-0002: 4 axes (grammar/creativity/moral_clarity/prompt_adherence) + objective metrics (perplexity, Distinct-1/2, Self-BLEU, Flesch) + 3-family judge panel + Cohen's κ / Kendall's τ; conclude by rank. Cite Nadas et al. (2025) arXiv:2504.20605; scaling grounding Kaplan (2020), Hoffmann (2022), Muennighoff (2023); KD Hinton (2015).
- Registry ids: keep `slm-30m`; add `slm-10m-distilled` (kind `scratch-slm`).
- TF1 fields: `prompt`, `fable`, `prompt_hash`. Existing interfaces: `scripts/tf1_pretrain/parse.parse_slots`, `scripts/tf1_pretrain/format.{build_training_text,apply_dropout,length_bucket,SEP,END,PAD}`.

---

## File Structure

- `scripts/prepare_tf1_pretrain.py` — MODIFY: add fable word-count quality filter + CLI args; default train-n 600k.
- `scripts/distill.py` — CREATE: pure `distillation_loss(...)` (torch), unit-tested.
- `notebooks/pretrain_slm_tf1.ipynb` — MODIFY: full-Chinchilla 30M HYPERPARAMS + Drive checkpoint/resume; add a distillation section (cells) that trains the 10M student from the 30M teacher.
- `config/models.json` — MODIFY: add `slm-10m-distilled`.
- Tests under `tests/`.

---

## Task 1 (S1): Data quality filter + larger subset

**Files:**
- Modify: `scripts/prepare_tf1_pretrain.py`
- Test: `tests/test_prepare_tf1_pretrain.py` (extend)

**Interfaces:**
- Consumes: `parse_slots`, `apply_dropout`, `build_training_text`, `length_bucket` (unchanged).
- Produces: `build_record(row, rng, min_words=0, max_words=None) -> dict | None` (None when the fable is empty OR its word count is `< min_words` or `> max_words`); `_write_split(rows, out_path, n, seed, min_words=0, max_words=None)`; CLI adds `--min-words`, `--max-words`, and default `--train-n 600000`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_prepare_tf1_pretrain.py
import random
from scripts.prepare_tf1_pretrain import build_record

def _row(fable):
    return {"prompt": "- Main Character: a fox\n- Teaching: be kind\n",
            "fable": fable, "prompt_hash": "h1"}

def test_build_record_rejects_too_short():
    # 3-word fable, min_words=5 -> rejected
    assert build_record(_row("A short tale."), random.Random(0), min_words=5) is None

def test_build_record_rejects_too_long():
    long_fable = " ".join(["word"] * 400)
    assert build_record(_row(long_fable), random.Random(0), max_words=320) is None

def test_build_record_accepts_in_range():
    ok_fable = " ".join(["word"] * 100) + " moral: be kind."
    rec = build_record(_row(ok_fable), random.Random(0), min_words=60, max_words=320)
    assert rec is not None and "text" in rec and "cond_len" in rec
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_prepare_tf1_pretrain.py -k "reject or in_range" -v`
Expected: FAIL (build_record has no min_words/max_words params → TypeError).

- [ ] **Step 3: Implement the filter**

Replace `build_record` and thread the params through `_write_split` and `main`:

```python
def build_record(row: dict, rng: random.Random,
                 min_words: int = 0, max_words: int | None = None) -> dict | None:
    fable = (row.get("fable") or "").strip()
    if not fable:
        return None
    wc = len(fable.split())
    if wc < min_words:
        return None
    if max_words is not None and wc > max_words:
        return None
    slots = parse_slots(row.get("prompt") or "")
    slots = apply_dropout(slots, rng)
    length = length_bucket(wc)
    text, cond_len = build_training_text(slots, fable, length)
    return {"text": text, "cond_len": cond_len}


def _write_split(rows, out_path: Path, n: int, seed: int,
                 min_words: int = 0, max_words: int | None = None) -> int:
    rng = random.Random(seed)
    seen: set[str] = set()
    written = 0
    with out_path.open("w") as f:
        for row in rows:
            h = row.get("prompt_hash")
            if h in seen:
                continue
            seen.add(h)
            rec = build_record(row, rng, min_words, max_words)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= n:
                break
    return written
```

In `main`, add args + pass through:

```python
    ap.add_argument("--train-n", type=int, default=600_000)
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=320)
    ...
    n_tr = _write_split(train, out / "train.jsonl", args.train_n, args.seed,
                        args.min_words, args.max_words)
    ...
    n_te = _write_split(test, out / "test.jsonl", args.test_n, args.seed + 1,
                        args.min_words, args.max_words)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_prepare_tf1_pretrain.py -v`
Expected: PASS (existing + 3 new tests). The `datasets` import stays lazy inside `main`, so tests need no heavy deps.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_tf1_pretrain.py tests/test_prepare_tf1_pretrain.py
git commit -m "feat: fable word-count quality filter + 600k default subset"
```

---

## Task 2 (S2): Teacher 30M full-Chinchilla config + Drive checkpoint/resume

**Files:**
- Modify: `notebooks/pretrain_slm_tf1.ipynb`

**Interfaces:**
- Produces: an updated notebook whose 30M run targets ~600M tokens and checkpoints to Drive. Later tasks/distillation load the teacher from `DRIVE/30M` (an HF model dir) and the corpus from `data/tf1`.

Notebook edits (this is a Colab notebook; the full training RUN is interactive on T4 — verify the notebook parses + is internally consistent, defer the multi-hour run). Use `nbformat` to edit; keep markdown/section structure.

- [ ] **Step 1: Update the HYPERPARAMETERS cell for full-Chinchilla**

In the `# =========================== TRAINING HYPERPARAMETERS ===` cell, set the v2 budget with comments:

```python
PEAK_LR      = 2e-3        # v2: lowered from 3e-3 for stability over a long (~8k-step) run
BATCH_SIZE   = 32          # sequences per forward/backward on the GPU
GRAD_ACCUM   = 8           # v2: effective batch = 32*8 = 256 sequences/update
STEPS        = 7900        # v2: ~600M tokens (Chinchilla 20x for 30M); ~4 epochs over 600k subset
LOG_EVERY    = 50
SAVE_EVERY   = 400         # v2: checkpoint to Drive every 400 steps (resume-safe)
```

(Keep `SEQ_LEN=512`, `ARCH`, `ADAM_BETAS`, `WEIGHT_DECAY=0.1`, `GRAD_CLIP=1.0`, `WARMUP_FRAC=0.02`, `DECAY_FRAC=0.20`, `FP16=True`.)

- [ ] **Step 2: Update `train(size)` to checkpoint + resume via Drive**

Change the `TrainingArguments` + `train()` in the model/train cell so output_dir is on Drive, checkpoints are saved, and training resumes if a checkpoint exists:

```python
    out_dir = f"{DRIVE}/{size}"                      # DRIVE = /content/drive/MyDrive/slm_tf1
    os.makedirs(out_dir, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir, max_steps=STEPS, fp16=FP16,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=GRAD_CLIP, logging_steps=LOG_EVERY,
        save_steps=SAVE_EVERY, save_total_limit=3,
        lr_scheduler_type="constant", report_to=[],   # WSD applied via LambdaLR
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS, data_collator=collator,
                      optimizers=(optimizer, scheduler))
    ckpts = [d for d in os.listdir(out_dir) if d.startswith("checkpoint-")] if os.path.isdir(out_dir) else []
    trainer.train(resume_from_checkpoint=bool(ckpts))
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    p = f"{out_dir}/tokenizer_config.json"; c = json.load(open(p))
    c["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(c, open(p, "w"))
    print(f"[{size}] saved -> {out_dir}")
```

Note: the WSD `LambdaLR` counter restarts on resume; acceptable for this project (loss curve continues to fall). Document this in a comment in the cell.

- [ ] **Step 3: Update Step-5 markdown + train call**

The train-both markdown/cells: keep only the 30M call for v2 (the 10M now comes from distillation, Task 3). Set the "Step 5" markdown to note 30M ~600M tokens ≈ several hours on T4, resumable. The prepare cell (Step 1) already accepts the new defaults; set its command to `--train-n 600000 --min-words 60 --max-words 320`.

```python
train("30M")   # v2: full Chinchilla ~600M tokens; resumable via Drive checkpoints
```

- [ ] **Step 4: Verify the notebook parses + is consistent (no GPU)**

Run:
```bash
python3 -c "import nbformat; nb=nbformat.read('notebooks/pretrain_slm_tf1.ipynb',as_version=4); print(len(nb.cells),'cells'); src=chr(10).join(c.source for c in nb.cells); assert 'STEPS        = 7900' in src or 'STEPS' in src and '7900' in src; assert 'save_steps' in src and 'resume_from_checkpoint' in src; [compile(c.source,'<c>','exec') for c in nb.cells if c.cell_type=='code' and 'pip' not in c.source and '!' not in c.source and 'drive' not in c.source]; print('parse+compile OK')"
```
Expected: prints cell count + `parse+compile OK`; asserts the 7900-step budget, save_steps, and resume wiring are present.

- [ ] **Step 5: Commit**

```bash
git add notebooks/pretrain_slm_tf1.ipynb
git commit -m "feat: 30M full-Chinchilla config + Drive checkpoint/resume (v2)"
```

- [ ] **Step 6: Interactive run (tracked in ledger, not a code change)**

On Colab T4: mount Drive, run Steps 1-6, train 30M to ~600M tokens (resume across sessions as needed), export GGUF `slm-30m`. Executed during Phase S4 integration.

---

## Task 3 (S3): Knowledge-distillation loss + notebook distillation cell

**Files:**
- Create: `scripts/distill.py`
- Test: `tests/test_distill.py`
- Modify: `notebooks/pretrain_slm_tf1.ipynb` (add a distillation section)

**Interfaces:**
- Produces: `distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5, ignore_index=-100) -> torch.Tensor` (scalar). `student_logits`/`teacher_logits`: `[N, V]`; `labels`: `[N]` with `ignore_index` on masked (conditioning/pad) positions. KD computed only over non-ignored positions; returns 0 when all positions are ignored.

- [ ] **Step 1: Write the failing test** (`pytest.importorskip("torch")` so it skips cleanly if torch absent)

```python
# tests/test_distill.py
import pytest
torch = pytest.importorskip("torch")
from scripts.distill import distillation_loss

def test_all_ignored_returns_zero():
    s = torch.randn(3, 5); t = torch.randn(3, 5)
    y = torch.full((3,), -100)
    assert float(distillation_loss(s, t, y)) == 0.0

def test_matching_teacher_student_loss_is_small():
    # identical logits + labels = argmax -> KD term ~0, CE small
    logits = torch.tensor([[10.0, 0, 0], [0, 10.0, 0]])
    y = torch.tensor([0, 1])
    loss = float(distillation_loss(logits, logits.clone(), y, T=2.0, alpha=0.5))
    assert loss < 0.1

def test_alpha_blends_kd_and_ce():
    s = torch.randn(4, 6); t = torch.randn(4, 6); y = torch.randint(0, 6, (4,))
    only_ce = float(distillation_loss(s, t, y, alpha=0.0))
    blended = float(distillation_loss(s, t, y, alpha=0.5))
    assert only_ce >= 0 and blended >= 0 and blended != only_ce
```

- [ ] **Step 2: Run to verify it fails**

Run: `pip install torch` (CPU build; needed to actually run) then `python3 -m pytest tests/test_distill.py -v`
Expected: FAIL (module missing). If torch cannot be installed here, the tests SKIP — say so explicitly; do not fake a pass.

- [ ] **Step 3: Implement**

```python
# scripts/distill.py
"""Token-level knowledge-distillation loss (Hinton et al., 2015).

Student learns the teacher's softened next-token distribution (KL on logits)
plus the hard-label cross-entropy. Used to distill the 30M teacher into a 10M
student on story tokens only (conditioning positions are ignore_index).
"""
import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels,
                      T: float = 2.0, alpha: float = 0.5,
                      ignore_index: int = -100):
    mask = labels != ignore_index
    if int(mask.sum()) == 0:
        return student_logits.sum() * 0.0          # keeps graph, value 0
    s = student_logits[mask]
    t = teacher_logits[mask]
    y = labels[mask]
    kd = F.kl_div(F.log_softmax(s / T, dim=-1),
                  F.softmax(t / T, dim=-1),
                  reduction="batchmean") * (T * T)
    ce = F.cross_entropy(s, y)
    return alpha * kd + (1.0 - alpha) * ce
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_distill.py -v`
Expected: PASS (3 tests) when torch is installed.

- [ ] **Step 5: Commit**

```bash
git add scripts/distill.py tests/test_distill.py
git commit -m "feat: token-level knowledge-distillation loss"
```

- [ ] **Step 6: Add the distillation section to the notebook**

Add a markdown cell + code cell after the 30M training. The code cell (Colab, torch) loads the frozen 30M teacher, builds a fresh 10M student, and trains the student with `distillation_loss`. Complete cell:

```python
# === Distill 30M teacher -> 10M student (token-level KD) ===
import torch
from scripts.distill import distillation_loss
from transformers import LlamaConfig, LlamaForCausalLM

teacher = LlamaForCausalLM.from_pretrained(f"{DRIVE}/30M").to("cuda").eval()
for p in teacher.parameters(): p.requires_grad_(False)

student = LlamaForCausalLM(LlamaConfig(
    vocab_size=tok.vocab_size, max_position_embeddings=SEQ_LEN,
    tie_word_embeddings=True, **ARCH["10M"])).to("cuda")
print("student params(M):", round(sum(p.numel() for p in student.parameters())/1e6, 1))

KD_STEPS = 2600           # ~200M tokens (~1.5 epochs over the 600k subset)
opt = torch.optim.AdamW(student.parameters(), lr=3e-3, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
def wsd(s):
    w, d = int(0.02*KD_STEPS), int(0.2*KD_STEPS)
    return s/max(1,w) if s < w else (max(0.,(KD_STEPS-s)/max(1,d)) if s > KD_STEPS-d else 1.)
sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd)

from torch.utils.data import DataLoader
loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
student.train(); step = 0
scaler = torch.cuda.amp.GradScaler(enabled=FP16)
while step < KD_STEPS:
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=FP16):
            with torch.no_grad():
                t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            s_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            # shift for next-token prediction; flatten
            sl = s_logits[:, :-1].reshape(-1, s_logits.size(-1))
            tl = t_logits[:, :-1].reshape(-1, t_logits.size(-1))
            yl = batch["labels"][:, 1:].reshape(-1)
            loss = distillation_loss(sl, tl, yl, T=2.0, alpha=0.5)
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % LOG_EVERY == 0: print(f"kd step {step}/{KD_STEPS} loss {loss.item():.3f}")
        if step >= KD_STEPS: break

out_dir = f"{DRIVE}/10M-distilled"; import os; os.makedirs(out_dir, exist_ok=True)
student.save_pretrained(out_dir); tok.save_pretrained(out_dir)
import json as J; p=f"{out_dir}/tokenizer_config.json"; c=J.load(open(p)); c["tokenizer_class"]="PreTrainedTokenizerFast"; J.dump(c, open(p,"w"))
print("distilled student saved ->", out_dir)
```

Also extend the export cell to also convert `10M-distilled` -> `{DRIVE}/slm-10m-distilled.gguf` + `Modelfile-10M-distilled` (same TEMPLATE + params).

- [ ] **Step 7: Verify notebook parses (no GPU)**

Run:
```bash
python3 -c "import nbformat; nb=nbformat.read('notebooks/pretrain_slm_tf1.ipynb',as_version=4); src=chr(10).join(c.source for c in nb.cells); assert 'distillation_loss' in src and '10M-distilled' in src; print('distill cell present; cells:', len(nb.cells))"
```
Expected: prints confirmation.

- [ ] **Step 8: Commit**

```bash
git add notebooks/pretrain_slm_tf1.ipynb
git commit -m "feat: notebook distillation cell (30M teacher -> 10M student) + GGUF export"
```

- [ ] **Step 9: Interactive run (ledger)** — on Colab: after the 30M teacher exists, run the distillation cell + export. Deferred to S4.

---

## Task 4 (S4): Register distilled model + eval/integration

**Files:**
- Modify: `config/models.json`
- Test: `tests/test_api_en.py` (extend)

This registers the distilled student and runs the real train→distill→eval pipeline (interactive). Registry edit is code; the runs are tracked in the ledger.

**Interfaces:**
- Consumes: `slm-30m`, `slm-10m-distilled` GGUFs created on Colab + `ollama create`d locally; `scripts/eval_slm.py` (unchanged) with `--models slm-10m-distilled slm-30m qwen3-4b`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_api_en.py
def test_models_includes_distilled_slm():
    r = client.get("/models")
    by_id = {m["id"]: m for m in r.json()}
    assert "slm-10m-distilled" in by_id
    assert by_id["slm-10m-distilled"]["kind"] == "scratch-slm"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_api_en.py::test_models_includes_distilled_slm -v`
Expected: FAIL (id absent).

- [ ] **Step 3: Add the registry entry**

Append to `config/models.json` (after the existing `slm-30m` entry; ensure valid JSON commas):

```json
{"id": "slm-10m-distilled", "name": "Fable-SLM 10M (distilled)", "ollama": "slm-10m-distilled", "kind": "scratch-slm", "desc": "10M SLM distilled from the 30M teacher (token-level KD)"}
```

- [ ] **Step 4: Run to verify it passes + full suite**

Run: `python3 -m pytest tests/test_api_en.py::test_models_includes_distilled_slm -v && python3 -m pytest -q`
Expected: PASS; full suite stays green.

- [ ] **Step 5: Commit + run pipeline**

```bash
git add config/models.json tests/test_api_en.py
git commit -m "feat: register distilled 10M SLM in model registry"
```

Then interactively (tracked in ledger): run the notebook (prepare 600k, train 30M to ~600M tokens, distill to 10M, export GGUFs to Drive); download + `ollama create slm-30m` and `slm-10m-distilled` on the Mac; `ollama pull gemma2:2b llama3.2:3b`; run `python3 -m scripts.eval_slm --models slm-10m-distilled slm-30m qwen3-4b --limit 100` to produce `results/eval_summary.json`; verify Compare + Results in the app.

---

## Self-Review

**Spec coverage:**
- §3 data v2 (quality filter, 600k) → Task 1. ✅
- §4 teacher 30M full-Chinchilla (600M tokens, WSD, checkpoint/resume) → Task 2. ✅
- §5 student 10M distillation (token-level KD, T/alpha, mask) → Task 3. ✅
- §6 eval (panel + κ/τ) → Task 4 (reuses `scripts/eval_slm.py`, unchanged). ✅
- §7 integration (GGUF fixes + TEMPLATE, registry, Compare/Results) → Task 2/3 export cells + Task 4. ✅
- §8 phases S1-S4 → Tasks 1-4. ✅

**Placeholder scan:** The notebook edits (Task 2/3) and the interactive runs (Task 2 Step 6, Task 4 Step 5) are explicitly interactive-on-Colab, not vague placeholders — they have concrete cell code + verification commands. The testable units (quality filter, KD loss, registry) have complete code + tests.

**Type consistency:** `build_record(row, rng, min_words, max_words)` signature matches its test + `_write_split` call. `distillation_loss(student_logits, teacher_logits, labels, T, alpha, ignore_index)` matches its test + the notebook cell call. Registry id `slm-10m-distilled` matches the test + notebook export name + eval `--models`. Special tokens/format reuse v1 (`SEP`/`END`, `build_training_text`).

---

## Notes for the executor

- Tasks 1, 3 are local TDD (Task 3 needs `pip install torch` to actually run its tests; else it SKIPs — report which).
- Tasks 2, 3(notebook), and the Task 4 runs require Colab T4 + Ollama and are interactive; verify notebook parses locally, defer the multi-hour runs.
- `data/`, `*.gguf`, `out/`, model dirs are git-ignored; artifacts persist on Google Drive.
