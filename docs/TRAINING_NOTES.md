# Training Notes — English Fable Generator (tinystory-vn)

Comprehensive record of the training pipeline, decisions, and results for the
from-scratch fable transformer. Prepared for the IT5410 course report.

---

## 1. Project Overview

**Goal:** Train a small transformer from scratch (no fine-tuning of a base model)
to generate children's fables (ages 4–7) in English, conditioned on keyword seeds
(main character + moral lesson). The model is served locally via MLX on Apple
Silicon and integrated into a FastAPI + React web app with guardrails, streaming,
and evaluation.

**Constraint:** The course prohibits fine-tuning existing models. The project
must train a model from scratch. Decision recorded in ADR-0003
(`docs/adr/0003-from-scratch-200m.md`).

**Dataset:** `klusai/ds-tf1-en-3M` (TF1-EN-3M) — 3M synthetic moral fables
(CC BY 4.0). Paper: Nadas et al., 2025 (arXiv:2504.20605).

---

## 2. Architecture & Training Config

### 2.1. Model Architecture (GPT2-style, from scratch)

| Parameter | Value |
|---|---|
| Architecture | GPT2LMHeadModel (via `transformers`) |
| n_embd | 768 |
| n_layer | 7 |
| n_head | 12 |
| n_positions (block_size) | 1024 |
| **Total params** | **63.0M** |
| Activation | gelu_new |
| Dropout | 0.0 (all: resid, embd, attn) |
| tie_word_embeddings | True |

### 2.2. Tokenizer (BPE with Metaspace)

| Parameter | Value |
|---|---|
| Type | BPE (via `tokenizers` library) |
| Vocab size | 16,384 |
| Pre-tokenizer | Metaspace (replacement=▁, prepend_scheme=always) |
| Decoder | Metaspace (same config) |
| Special tokens | `<char>`, `</char>`, `<moral>`, `</moral>`, `<story>`, `</story>` |

**Why Metaspace:** The initial tokenizer (v1) used raw BPE with no pre-tokenizer.
This caused broken words in output ("par ty", "w o ven") because spaces were
treated as regular characters and the decoder inserted visible spaces between
subword fragments. Metaspace (same approach as LLaMA/T5) encodes spaces as a
sentinel character (▁) before BPE, so word boundaries survive merges and decode
cleanly. Verified: `"The clever fox threw a party to celebrate."` → lossless
round-trip.

### 2.3. Training Data

| Parameter | Value |
|---|---|
| Source | TF1-EN-3M (streaming via HuggingFace `datasets`) |
| Subset size | 200,000 fables |
| Filtering | English only, must have parseable character + moral, story ≥ 80 chars |
| Format | `<char> {character} </char>\n<moral> {moral} </moral>\n<story>\n{story}\n</story>` |
| Split | 95% train / 5% test (random, seed=42) |

**Data prep pipeline:** `scripts/prepare_tf1.py` streams TF1-EN-3M → filters valid
records → trains BPE tokenizer → writes `fables.jsonl` (formatted text) +
`tokenizer.json`. Runs locally on the Mac (stable network) before uploading to
Colab (avoids HF streaming timeouts / rate limits on the VM).

### 2.4. Training Hyperparameters

| Parameter | Value |
|---|---|
| Framework | `transformers.Trainer` + `GPT2LMHeadModel` |
| Batch size | 64 (per_device_train_batch_size=64, no grad accumulation) |
| Max steps | 15,625 (= 5 epochs × 3,125 steps/epoch) |
| Learning rate | 5e-4 |
| LR scheduler | Cosine |
| Warmup steps | 1,562 (max_steps // 10) |
| Weight decay | 0.1 |
| Precision | bf16 |
| Logging steps | 50 |
| Checkpoint saves | Every 500 steps (for resume) |
| Seed | 42 |
| GPU | NVIDIA A100 (Colab) |

---

## 3. Training Infrastructure

### 3.1. Colab via google-colab-cli

Training runs on Google Colab, orchestrated entirely from the terminal via
`google-colab-cli` (no browser interaction needed). Runbook:
`docs/runbooks/colab-train.md`.

**Workflow:**
1. `colab new -s a100 --gpu A100` — provision a paid A100 VM
2. Upload prepared data (chunked, since `colab upload` has a ~40MB limit per file)
3. Upload scripts (`train_local.py`, `metrics.py`, `prepare_tf1.py`,
   `fable_tokenizer.py`)
4. Launch training detached (`nohup`) so the CLI returns immediately
5. Poll progress via `colab download` of the training log
6. Download checkpoint + eval results
7. `colab stop -s a100` — release the VM (avoid idle billing)

### 3.2. Resume & Checkpoint Safety

Colab VMs can be reclaimed at any time (idle timeout, session caps). To protect
against losing training progress:

- **Checkpoint every 500 steps** to `/content/fable200m/ckpt/checkpoint-{step}/`
  (includes model weights, optimizer state, RNG state, scheduler — full resume
  state).
- **`resume_from_checkpoint=True`** in `trainer.train()` — auto-detects the
  latest checkpoint and resumes (step, optimizer, scheduler, RNG). Only resumes
  if a checkpoint exists (avoids `ValueError` on fresh start).
- **Background sync loop** (on the Mac): every 5 min, downloads the latest
  checkpoint from the VM via `colab download`, uploads to Google Drive via `gws`
  (Google Workspace CLI). If the VM is reclaimed, the latest checkpoint survives
  on Drive.
- **Recovery procedure:** re-provision A100 → download checkpoint from Drive
  (chunked) → reassemble on VM → relaunch training (auto-resumes from last step).

**Proven:** The 29M training run was reclaimed at step ~2000, recovered from
Drive checkpoint-2000, resumed, and completed to step 9375 successfully.

### 3.3. Data Staging

Data is prepared **locally on the Mac** (stable network, HF token for rate
limits), not on the Colab VM. The 313MB `fables.jsonl` is split into 8 × 40MB
chunks for `colab upload` (which rejects files >~40MB). A reassembly script
on the VM concatenates them back.

---

## 4. Training History & Iterations

### 4.0. Failed Attempt 1: 211M on T4 (never completed)

| Setting | Value |
|---|---|
| Date | 2026-07-15 |
| n_embd / n_layer / n_head | 1024 / 16 / 16 |
| Params | 211.0M |
| Vocab size | 8,192 |
| Tokenizer | BPE (no pre-tokenizer) |
| Max steps | 30,000 (arbitrary, ~9.6 epochs) |
| Batch | 16 (per_device), grad_accum 4 (effective 64) |
| GPU | T4 (free Colab) |
| Status | **FAILED** — multiple failure modes |

**What happened (sequential failures):**

1. **OOM kill (memory cgroup):** `iter_tf1()` buffered `n×4 = 800,000` raw rows
   into a list before filtering. On the T4 VM (~12 GB RAM), this exceeded the
   memory cgroup limit → process killed by OOM. `dmesg` confirmed:
   `Memory cgroup out of memory: Killed process (python3) total-vm:22GB`.
   **Fix:** Rewrote `iter_tf1()` to stream-filter on the fly, keeping only `n`
   valid records (commit `939eefd`).

2. **HF streaming stall (unauthenticated):** Without an `HF_TOKEN`, HuggingFace
   streaming was rate-limited. The process hung on data prep for 25+ min with no
   progress. **Fix:** Staged data locally on the Mac with an HF token, uploaded
   prepared files to Colab (no streaming on the VM).

3. **VM reclaimed during data prep:** Even with the token, the long data-prep
   phase (~25 min of HF streaming on the VM) caused Colab to reclaim the session
   before training even started. Session terminated at 17:05.
   **Fix:** Moved data prep entirely off the VM (local Mac → chunked upload).

4. **Drive mount hangs headlessly:** `drive.mount()` and
   `auth.authenticate_user()` require interactive OAuth (browser prompt) which
   hangs in a headless `colab exec` call. The kernel got stuck BUSY and had to
   be restarted.
   **Fix:** Used `gws` (Google Workspace CLI) from the Mac to manage Drive
   files, and `colab download`/`colab upload` for VM file transfer instead of
   Drive mount.

5. **`colab upload` file size limit (~40MB):** The 313MB `fables.jsonl` was
   rejected with 400/500 errors. **Fix:** Split into 8 × 40MB chunks, upload
   individually, reassemble on the VM.

6. **`colab exec` client timeout:** Long-running `colab exec` calls (notebook
   execution, detached training) hit the client's 10–30s read timeout when the
   VM was busy. The kernel kept running but the client gave up.
   **Fix:** Used detached `nohup` launch + `colab download` of the log file for
   monitoring instead of blocking `colab exec`.

**Probe results (before abandoning T4):**
- 211M at batch 16 on T4: ~0.5 s/step (2.0 it/s) → 30k steps ≈ 17 hours.
- Too slow for a free T4 (12h session cap + reclaim risk).
- GPU was ~14% utilized (MFU) — heavily underutilized at batch 16.

**Key decision:** Abandoned 211M on T4. Switched to paid A100 + smaller model.

### 4.0.1. A100 Probe (211M, 100 steps)

| Setting | Value |
|---|---|
| Params | 211.0M |
| Batch | 16 |
| Steps | 100 (probe only) |
| GPU | A100 |
| Speed | ~2.1 it/s (0.47 s/step) |
| GPU utilization | ~14% MFU (still underutilized at batch 16) |

**Finding:** A100 was ~4× faster than T4 but still GPU-underutilized. Bumping
batch size (16→64) would help more than shrinking the model. But 211M at
30k steps was still ~4h on A100 — decided to go smaller for a first runnable
model.

### 4.0.2. Decision: TinyStories approach (smaller model)

Researched the TinyStories paper (Eldan & Li, 2023): small models (10–33M) can
generate coherent children's stories with enough data and training. Decided to
drop from 211M to ~30M for a cheap first model + eval, matching the TinyStories
paradigm. This is documented in the conversation and influenced the v1 config.

### 4.1. Run 1: 29.9M model (v1, broken tokenizer)

| Setting | Value |
|---|---|
| Date | 2026-07-16 |
| n_embd / n_layer / n_head | 512 / 8 / 8 |
| Params | 29.9M |
| Vocab size | 8,192 |
| Tokenizer | BPE (no pre-tokenizer) ← **BUG** |
| Epochs | 3 (9,375 steps) |
| Batch | 64 |
| GPU | A100 |
| Training time | ~54 min (2.9 it/s) |
| Loss | 8.08 → 1.87 |
| Status | Completed, eval generated |

**Results:**
- distinct_1: 0.389, distinct_2: 0.857, self_bleu: 0.078, flesch: 82.9
- Output was coherent narrative but had **broken words** ("par ty", "w o ven
  ger") due to the tokenizer bug (no pre-tokenizer → spaces inserted between
  subword fragments).
- Checkpoint saved to Google Drive (`fable200m/run1/`).

**Issues identified:**
1. **Tokenizer bug:** BPE with no pre-tokenizer produced broken words.
2. **Repetition:** Model repeated phrases ("par ty had come" looped).
3. **Small vocab:** 8,192 caused excessive word splitting.
4. **Small model:** 29.9M produced coherent but sometimes rambling output.

### 4.2. Run 2: 63M model (v2, fixed tokenizer) — IN PROGRESS

| Setting | Value |
|---|---|
| Date | 2026-07-16 |
| n_embd / n_layer / n_head | 768 / 7 / 12 |
| Params | 63.0M |
| Vocab size | 16,384 |
| Tokenizer | BPE + Metaspace pre-tokenizer/decoder ← **FIXED** |
| Epochs | 5 (15,625 steps) |
| Batch | 64 |
| GPU | A100 |
| Est. speed | ~1.68 it/s |
| Est. time | ~2.6 hours |
| Status | Training (step ~1153/15625 at time of writing) |

**Early loss curve (v2):**
```
epoch 0.27: loss 3.252
epoch 0.29: loss 3.150
epoch 0.30: loss 3.062
epoch 0.32: loss 2.972
epoch 0.34: loss 2.916
```

**Improvements over v1:**
- Metaspace tokenizer (clean word boundaries, no broken words)
- 2× vocab (16,384 → fewer word splits)
- 2.1× params (63M vs 30M → better narrative capacity)
- 1.67× epochs (5 vs 3 → better convergence)
- Shorter generation (150 tokens max → less rambling)

---

## 5. Bugs Found & Fixed

### 5.1. OOM on Colab (T4, 12GB RAM)

**Bug:** `iter_tf1()` buffered `n×4 = 800,000` raw rows into a list before
filtering, exceeding the VM's memory cgroup → OOM kill.

**Fix:** Stream-filter rows on the fly, keeping only `n` valid records in memory
(`scripts/prepare_tf1.py`, commit `939eefd`).

### 5.2. VM Reclaim with No Persistence

**Bug:** Training wrote checkpoints to `/content/` (ephemeral VM disk). When the
VM was reclaimed, all progress was lost.

**Fix:**
1. Added `resume_from_checkpoint=True` to `trainer.train()` (commit `9b85675`).
2. Fixed `ValueError` on fresh start (no checkpoint → don't resume, commit
   `8861c71`).
3. Built a background sync loop: VM checkpoint → Mac → Google Drive (via `gws`).
4. Recovery procedure documented and proven.

### 5.3. Tokenizer: Broken Words

**Bug:** BPE trained with `Tokenizer(BPE())` and no pre-tokenizer. Spaces treated
as regular characters → decoded text had visible spaces between subword fragments
("par ty", double spaces).

**Fix:** Added `Metaspace` pre-tokenizer + decoder (commit `a721048`). Verified
lossless round-trip: `"The clever fox threw a party to celebrate."` → tokens →
decoded back identically.

### 5.4. HF→MLX Weight Conversion

**Bug:** `mlx_lm.convert` failed because:
1. MLX's GPT2 `ModelArgs` requires `n_ctx` (HF config has `n_positions`).
2. HF keys use `transformer.h.0...` but MLX expects `model.h.0...`.
3. HF Conv1D stores weights as `(in, out)` but MLX Linear expects `(out, in)`.
4. `bos_token_id`/`eos_token_id` defaulted to GPT2's 50256 (out of our 8192
   vocab range).

**Fix:** Manual conversion script:
- Added `n_ctx` to config.
- Remapped keys: `transformer.*` → `model.*`.
- Transposed Linear weight matrices (c_attn, c_proj, c_fc, c_proj/mlp).
- Fixed bos/eos to match BPE vocab (4=`<story>`, 5=`</story>`).

### 5.5. MLX Server Model ID Mismatch

**Bug:** MLX server identifies models by absolute path. App sent a short name
(`fable-29m-mlx`) → 404.

**Fix:** `models_registry.py` now auto-detects the model ID by querying the
server's `/v1/models` endpoint (commit `b2e7b75`).

### 5.6. Chat Format vs Base LM

**Bug:** The app's `/v1/chat/completions` endpoint sent chat-formatted messages
(role tokens) to a base LM that was trained on raw text continuation → degenerate
output.

**Fix:** Added `/v1/completions` (raw text completion) support with
`repetition_penalty` parameter (commit `56569c8`). Controlled by
`FABLE_USE_COMPLETION=true` (default).

---

## 6. Local Serving (MLX)

### 6.1. MLX Server

The trained model is served locally on the M4 Pro via `mlx-lm`'s built-in
server:

```bash
uv run python -m mlx_lm server --model models/fable-64m-mlx --port 8080
```

- **Speed:** ~1,292 tok/s generation on M4 Pro (MLX is native Apple Silicon).
- **API:** OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`,
  `/v1/models`).

### 6.2. App Integration

The app's `ollama_client.py` supports two backends:
- `FABLE_BACKEND=ollama` (default): Ollama's `/api/chat` (for the base Qwen3
  model).
- `FABLE_BACKEND=openai`: OpenAI-compatible API. Uses `/v1/completions` for base
  LMs (raw text continuation with `repetition_penalty`) or
  `/v1/chat/completions` for instruction-tuned models.

```bash
# Start MLX server
uv run python -m mlx_lm server --model models/fable-64m-mlx --port 8080

# Start app
FABLE_BACKEND=openai OLLAMA_BASE_URL=http://127.0.0.1:8080 \
  uv run python -m uvicorn app.main:app --port 8000
```

### 6.3. Model Registry

`config/models.json`:
```json
[
  {"id": "base-qwen3-4b", "name": "Qwen3-4B-Instruct-2507", "ollama": "qwen3-4b-instruct", "kind": "base", ...},
  {"id": "fable-200m", "name": "Fable-64M (from scratch)", "ollama": "fable-64m-mlx", "kind": "finetuned", ...}
]
```

Model name is auto-detected by querying the MLX server's `/v1/models` endpoint
(no hardcoded path).

---

## 7. Evaluation

### 7.1. Reference-Free Metrics (automated, in `scripts/metrics.py`)

| Metric | v1 (29M) | v2 (64M, pending) | Description |
|---|---|---|---|
| distinct_1 | 0.389 | TBD | Unigram diversity (higher = more diverse) |
| distinct_2 | 0.857 | TBD | Bigram diversity |
| self_bleu | 0.078 | TBD | Cross-sample repetition (lower = better) |
| flesch_reading_ease | 82.9 | TBD | Reading difficulty (higher = easier, 70+ = 6th grade) |

### 7.2. LLM-as-Judge (4-axis, per ADR-0002)

Not yet run for the from-scratch model. The app's `/evaluate` endpoint calls a
judge model to score 4 axes (1–10 scale):
- **Grammar** — grammatical correctness / coherence
- **Creativity** — creativity / engagement
- **Moral Clarity** — how clear and well-conveyed the moral is
- **Prompt Adherence** — adherence to user-provided narrative elements

Note: The from-scratch base model cannot serve as a judge (it's not
instruction-tuned). A separate instruct model (e.g. Qwen3-4B-Instruct via
Ollama, or SmolLM2-1.7B-Instruct on Colab) is needed for judging.

### 7.3. Batch Eval (Notebook B)

`notebooks/eval_gen_fable200m_colab.ipynb` loads the checkpoint on Colab,
generates fables from seed prompts, computes reference-free metrics +
LLM-as-judge scores, and writes `results/eval_summary.json` (consumed by the
app's Results tab).

---

## 8. Key Files

| File | Role |
|---|---|
| `scripts/prepare_tf1.py` | Stream TF1 → filter → train BPE → write fables.jsonl + tokenizer.json |
| `scripts/train_local.py` | Single-script: train → generate → metrics → eval_summary.json |
| `scripts/metrics.py` | Distinct-1/2, Self-BLEU, Flesch Reading Ease |
| `scripts/fable_tokenizer.py` | Char-level tokenizer (offline fallback) |
| `notebooks/train_fable200m_colab.ipynb` | Notebook A: train on Colab |
| `notebooks/eval_gen_fable200m_colab.ipynb` | Notebook B: eval + gen on Colab |
| `app/ollama_client.py` | LLM client (Ollama + OpenAI-compatible backends) |
| `app/models_registry.py` | Model registry + auto-detect MLX model ID |
| `app/config.py` | Env vars: BACKEND, OLLAMA_BASE_URL, GEN_* params, etc. |
| `app/prompt_en.py` | System prompts, length hints, build_fable_prompt + build_seed_prompt |
| `config/models.json` | Model registry (base + from-scratch) |
| `results/eval_summary.json` | Batch eval results (metrics + sample fables) |
| `docs/adr/0003-from-scratch-200m.md` | Decision: train from scratch, not fine-tune |
| `docs/adr/0002-evaluation-methodology.md` | Eval methodology (4-axis LLM-as-judge) |
| `docs/runbooks/colab-train.md` | Colab training runbook |

---

## 9. Commit History (training-related)

| Commit | Description |
|---|---|
| `a12ed28` | Remove stale VN/fine-tune artifacts, reflect Colab from-scratch training |
| `faa1da9` | Bump Colab notebook to full 200k/30k-step run, save to Drive |
| `939eefd` | Fix: stream-filter TF1 rows to avoid OOM (drop n×4 oversample) |
| `9b85675` | Train: support resume_from_checkpoint + frequent checkpoints |
| `8861c71` | Fix: resume only when checkpoint exists (avoid ValueError) |
| `97673fe` | Add first trained model eval results (fable-29M, 3 epochs) |
| `56569c8` | MLX backend — serve fable model locally via mlx-lm server |
| `f5e1fd3` | Fix: use full local path as model name for MLX server |
| `b2e7b75` | Fix: auto-detect MLX server model ID via /v1/models |
| `a721048` | Fix: BPE tokenizer with Metaspace (clean word boundaries) |
| `520d788` | Tune: shorter stories (short=150tok, medium=350tok, long=600tok) |

---

## 10. Hardware & Tools

| Tool | Version | Role |
|---|---|---|
| Python | 3.12 | Runtime (uv-managed venv) |
| uv | 0.11.19 | Package management |
| transformers | 5.14.0 | Training (GPT2LMHeadModel, Trainer) |
| tokenizers | 0.22.2 | BPE tokenizer training |
| mlx-lm | 0.31.3 | Local inference (MLX on Apple Silicon) |
| google-colab-cli | — | Colab VM provisioning + file sync |
| gws (Google Workspace CLI) | 0.16.0 | Google Drive file upload/download |
| FastAPI | — | Backend API |
| React + TypeScript | — | Frontend (Vite + Astryx + recharts) |
| pytest | — | Testing (65 tests) |

| Hardware | Role |
|---|---|
| Apple M4 Pro | Local dev + MLX inference |
| NVIDIA A100 (Colab) | Training |
| Google Drive | Checkpoint backup + data staging |

---

## 11. Lessons Learned

1. **Don't stream large datasets on Colab VMs.** HF streaming is rate-limited
   (unauthenticated) and the VM can be reclaimed mid-stream. Stage data locally
   first, upload prepared files.

2. **Always checkpoint to persistent storage.** Colab VMs are ephemeral. Use
   Drive (via `gws` from the Mac) as the persistence layer, with a background
   sync loop.

3. **Tokenizer pre-tokenizer matters.** Training a BPE without a pre-tokenizer
   produces broken words. Metaspace (LLaMA/T5 approach) is the right choice for
   clean word boundaries.

4. **Base LMs need completion, not chat.** The MLX server's `/v1/chat/completions`
   formats input with role tokens the model never saw. Use `/v1/completions`
   (raw text) for base LMs, with `repetition_penalty` to prevent loops.

5. **HF→MLX conversion requires key remapping + weight transpose.** GPT2's
   Conv1D stores `(in, out)`; MLX's Linear expects `(out, in)`. Keys need
   `transformer.` → `model.` prefix stripping.

6. **Small models are fast but need more data/steps, not bigger params.** The
   TinyStories paradigm: data scale > param scale for simple domains. 63M on
   200k fables × 5 epochs is a reasonable baseline.

---

## 12. Next Steps

1. **Complete v2 training** (63M, 5 epochs, Metaspace tokenizer) — in progress.
2. **Convert v2 checkpoint to MLX** — same key remap + transpose process.
3. **Run Notebook B** (LLM-as-judge eval) on Colab — 4-axis scores for the
   report.
4. **Test the full app** with the v2 model — Playground, Compare, Results.
5. **Consider scaling:** more data (500k fables), more epochs, or larger model
   (100M+) if quality is insufficient.
