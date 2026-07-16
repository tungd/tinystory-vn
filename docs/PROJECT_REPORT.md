# Project Report — tinystory-vn: English Fable Generator

## 1. Objective

Train a small transformer model **from scratch** (no fine-tuning) to generate
children's fables (ages 4–7) in English. The model is conditioned on keyword
seeds (main character + moral lesson) and served locally via MLX on Apple
Silicon, integrated into a FastAPI + React web app with guardrails, streaming,
and LLM-as-judge evaluation.

**Constraint:** The course prohibits fine-tuning existing models. All training
is from random initialization. (ADR-0003)

---

## 2. Dataset

**Source:** `klusai/ds-tf1-en-3M` (TF1-EN-3M) — 3M synthetic moral fables
(CC BY 4.0). Paper: Nadas et al., 2025 (arXiv:2504.20605).

**Subset used:** 200,000 fables (filtered for English, valid character + moral
fields, story ≥ 80 chars).

**Format:** Each fable is formatted as:
```
<char> {character} </char>
<moral> {moral} </moral>
<story>
{story text}
</story>
```

**Split:** 95% train / 5% test (seed=42).

---

## 3. Model Architecture

GPT2-style decoder-only transformer (via `transformers.GPT2LMHeadModel`):

| Parameter | Value |
|---|---|
| Architecture | GPT2LMHeadModel |
| n_embd | 768 |
| n_layer | 7 |
| n_head | 12 |
| n_positions | 1024 |
| **Total params** | **63.0M** |
| Activation | gelu_new |
| Dropout | 0.0 |
| tie_word_embeddings | True |

---

## 4. Tokenizer

BPE tokenizer trained from scratch via the `tokenizers` library:

| Parameter | Value |
|---|---|
| Type | BPE |
| Vocab size | 16,384 |
| Pre-tokenizer | Metaspace (▁ sentinel) |
| Decoder | Metaspace |
| Special tokens | `<char>`, `</char>`, `<moral>`, `</moral>`, `<story>`, `</story>` |

**Critical bug found and fixed:** The initial tokenizer (v1) used raw BPE with
no pre-tokenizer. This caused broken words in output ("par ty", "w o ven") because
spaces were treated as regular characters and the decoder inserted visible spaces
between subword fragments. Fixed by adding Metaspace pre-tokenizer/decoder (same
approach as LLaMA/T5), which encodes spaces as a sentinel character before BPE
and restores them on decode. Verified lossless round-trip.

---

## 5. Training Infrastructure

### 5.1. Colab via google-colab-cli

Training runs on Google Colab A100 GPUs, orchestrated from the terminal via
`google-colab-cli` (no browser interaction needed for training itself).

### 5.2. Google Drive Mount

Data, checkpoints, and eval results are stored on Google Drive, mounted on the
Colab VM via `colab drivemount`. This makes checkpoints **persistent** — they
survive VM reclaims automatically. The mount requires a one-time interactive
OAuth authorization (browser-based).

**Key insight:** Saving to the VM's ephemeral `/content/` disk was the root
cause of all checkpoint losses. Once we switched to Drive-mounted output, training
became resilient to reclaims.

### 5.3. Resume Support

`train_local.py` uses HuggingFace Trainer's `resume_from_checkpoint=True`, which
auto-detects the latest checkpoint in the output directory and resumes (optimizer
state, scheduler, RNG). Each checkpoint is a separate directory
(`checkpoint-{step}/`), so if one is corrupted by a mid-write reclaim, the
previous one is still intact (stepped backup).

### 5.4. Google Workspace CLI (gws)

Used for uploading data to Drive and downloading results from the Mac, without
the browser. Available at `/opt/local/bin/gws`.

---

## 6. Training History

### 6.1. Run 1: 29.9M model (v1, broken tokenizer)

| Setting | Value |
|---|---|
| n_embd / n_layer / n_head | 512 / 8 / 8 |
| Params | 29.9M |
| Vocab size | 8,192 |
| Tokenizer | BPE (no pre-tokenizer) — **BUG** |
| Epochs | 3 (9,375 steps) |
| Loss | 8.08 → 1.87 |
| Status | Completed |

**Results:**
- distinct_1: 0.389, distinct_2: 0.857, self_bleu: 0.078, flesch: 82.9
- Output had broken words ("par ty", "w o ven") due to tokenizer bug.

### 6.2. Run 2: 63M model (v2, fixed tokenizer)

| Setting | Value |
|---|---|
| n_embd / n_layer / n_head | 768 / 7 / 12 |
| Params | 63.0M |
| Vocab size | 16,384 |
| Tokenizer | BPE + Metaspace — **FIXED** |
| Epochs | 2 (6,250 steps) |
| Loss | 3.2 → 1.73 |
| Status | Completed |

**Results:**
- distinct_1: 0.519 (+33% vs v1)
- distinct_2: 0.922 (+8%)
- self_bleu: 0.028 (-64%, less repetitive)
- flesch: 81.5
- Clean word boundaries, coherent narrative

### 6.3. Comparison

| Metric | v1 (29M) | v2 (63M) | Change |
|---|---|---|---|
| distinct_1 | 0.389 | 0.519 | +33% |
| distinct_2 | 0.857 | 0.922 | +8% |
| self_bleu | 0.078 | 0.028 | -64% (better) |
| flesch | 82.9 | 81.5 | similar |
| Word quality | Broken | Clean | Fixed |

---

## 7. Failed Attempts & Lessons

### 7.1. OOM on Colab T4

`iter_tf1()` buffered 800k rows in RAM → OOM kill. Fixed by stream-filtering.

### 7.2. VM Reclaim with No Persistence

Training wrote to `/content/` (ephemeral). When VM was reclaimed, all progress
lost. Fixed by mounting Google Drive and writing checkpoints directly there.

### 7.3. Tokenizer Bug

BPE with no pre-tokenizer produced broken words. Fixed with Metaspace.

### 7.4. HF→MLX Conversion

Required key remapping (`transformer.*` → `model.*`), weight transpose
(Conv1D → Linear), and config fixes (`n_ctx`, `bos/eos_token_id`).

### 7.5. MLX Server Model ID Mismatch

MLX server identifies models by absolute path. Fixed by auto-detecting via
`/v1/models` endpoint in `models_registry.py`.

### 7.6. Chat Format vs Base LM

The MLX server's `/v1/chat/completions` formats input with role tokens the base
model never saw. Fixed by using `/v1/completions` (raw text completion) with
`repetition_penalty` support.

### 7.7. Drive Mount Requires Interactive OAuth

`colab drivemount` needs browser-based OAuth authorization. Cannot be done
fully headlessly. Once authorized, the mount persists for the session.

---

## 8. Local Serving (MLX)

The trained model is served locally on the M4 Pro via `mlx-lm`:

```bash
uv run python -m mlx_lm server --model models/fable-64m-mlx --port 8080
```

- **Speed:** ~1,292 tok/s on M4 Pro
- **API:** OpenAI-compatible (`/v1/completions`, `/v1/models`)

### HF→MLX Conversion

1. Key remap: `transformer.*` → `model.*`
2. Weight transpose: Conv1D `(in, out)` → Linear `(out, in)`
3. Config: add `n_ctx`, fix `bos/eos_token_id`

### App Integration

`app/ollama_client.py` supports two backends:
- `FABLE_BACKEND=ollama`: Ollama's `/api/chat` (for instruction-tuned models)
- `FABLE_BACKEND=openai`: OpenAI-compatible API, uses `/v1/completions` for base
  LMs with `repetition_penalty` support

---

## 9. Evaluation

### 9.1. Reference-Free Metrics (automated)

| Metric | v1 | v2 | Description |
|---|---|---|---|
| distinct_1 | 0.389 | 0.519 | Unigram diversity |
| distinct_2 | 0.857 | 0.922 | Bigram diversity |
| self_bleu | 0.078 | 0.028 | Cross-sample repetition (lower=better) |
| flesch | 82.9 | 81.5 | Reading ease (70+ = 6th grade) |

### 9.2. LLM-as-Judge (4-axis, via Google AI Studio)

Judge model: **Gemma 4 31B** (via Google AI Studio free tier, 15k RPD quota).

Four axes scored 1–10:
- **Grammar** — grammatical correctness
- **Creativity** — creativity / engagement
- **Moral Clarity** — how clear the moral is
- **Prompt Adherence** — adherence to requested character/moral

The judge uses a separate backend (`FABLE_JUDGE_BACKEND=openai`) with the
Google AI Studio API. Judge calls use `is_judge=True` to route to the correct
backend while generation uses MLX.

Tested end-to-end: MLX generates fable → Gemma 4 31B judges with evidence-cited
rationale.

---

## 10. Key Files

| File | Role |
|---|---|
| `scripts/prepare_tf1.py` | Stream TF1 → filter → train BPE → write fables.jsonl + tokenizer.json |
| `scripts/train_local.py` | Train → generate → metrics → eval_summary.json (single script) |
| `scripts/metrics.py` | Distinct-1/2, Self-BLEU, Flesch Reading Ease |
| `app/ollama_client.py` | LLM client (Ollama + OpenAI-compatible backends, per-call judge override) |
| `app/models_registry.py` | Model registry + auto-detect MLX model ID |
| `app/judge.py` | 4-axis LLM-as-judge prompt + parser |
| `app/config.py` | Env vars: BACKEND, JUDGE_BACKEND, GEN params |
| `config/models.json` | Model registry (base + fable + judge) |
| `results/eval_summary.json` | Batch eval results (metrics + sample fables) |
| `models/fable-64m-mlx/` | MLX-format 63M model (config + weights + tokenizer) |
| `.env` | Env config (gitignored, contains API keys) |

---

## 11. Commit History (key commits)

| Commit | Description |
|---|---|
| `939eefd` | Fix: stream-filter TF1 rows to avoid OOM |
| `9b85675` | Train: support resume_from_checkpoint + frequent checkpoints |
| `8861c71` | Fix: resume only when checkpoint exists |
| `97673fe` | Add first trained model eval results (fable-29M) |
| `56569c8` | MLX backend — serve fable model locally |
| `b2e7b75` | Fix: auto-detect MLX server model ID |
| `a721048` | Fix: BPE tokenizer with Metaspace (clean word boundaries) |
| `520d788` | Tune: shorter stories |
| `c93a959` | Prepare Gemini judge backend + skip length hints for base LM |
| `f4d9ebc` | Wire Gemma 4 31B as LLM-as-judge via Google AI Studio |
| `3855bde` | Tune: save_steps=250 for frequent checkpoints |
| `25e8ad5` | v2 model eval results (63M, Metaspace, 2 epochs) |
| `2c6b329` | 64M v2 model — MLX format, clean word boundaries |

---

## 12. Current State & Next Steps

### Done
- ✅ 63M model trained (2 epochs, loss 1.73, clean output)
- ✅ MLX serving working locally (~1,292 tok/s)
- ✅ App supports dual backends (MLX for generation, Gemma for judging)
- ✅ Drive mount workflow proven (checkpoints survive reclaims)
- ✅ Eval results committed

### Known Limitations
- Stories are somewhat short and don't always state the moral explicitly
- 2 epochs may not be enough for strong narrative coherence
- No explicit moral statement in training data format

### Next Steps
1. Train more epochs (resume from checkpoint-6250 → 9375, ~35 min)
2. Modify training data to append explicit moral statements
3. Run full 4-axis LLM-as-judge eval with the trained model
4. Test the full app end-to-end (Playground + Compare + Results)
