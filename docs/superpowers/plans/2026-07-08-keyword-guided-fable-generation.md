# Keyword-Guided Fable Generation — 200M nanoGPT (from scratch)

> **Approach:** Train a **~200M-param GPT2-style transformer from scratch** (via
> `transformers` + `tokenizers`, installed with `uv`) on a subset of TF1-EN-3M,
> conditioned on two keyword seeds — **main character** and **moral lesson** — via
> control-prefix tokens. No base model, no LoRA, no RAG. This is pure training (not
> fine-tuning), so it sidesteps the fine-tune restriction. (Implementation pivoted
> from hand-rolled nanoGPT to the `transformers.Trainer` + BPE path.)
>
> **Keyword guidance** = seed tokens prepended to each training/eval sequence:
> `<char> ... </char> <moral> ... </moral>\n<story> ... </story>`. At generation time the
> model is fed the seeds and decodes the fable. Pure autoregressive generation.

---

## 1. Goal

Train a small fable language model on Colab, orchestrated from the terminal via
`google-colab-cli`, that generates children's fables (ages 4–7) conditioned on
`(character, moral)` keyword seeds. Output: a downloadable weight checkpoint (`.pt` /
exported GGUF). A **second** Colab notebook loads the checkpoint for eval + story gen.
The local Ollama app registers the exported checkpoint for demo (Compare / Results).

This is the **TinyStories paradigm** (small models on simple kids' stories) applied to
TF1-EN-3M's 3M moral fables.

---

## 2. Model & data

- **Architecture (nanoGPT):** ~200M params — `n_layer≈12, n_head≈12, n_embd≈768,
  block_size≈1024`, learned BPE tokenizer (vocab ~8k) fit on the fable subset.
- **Data:** stream `klusai/ds-tf1-en-3m` (or load a local subset), keep records with a
  parseable `character` + `teaching/moral`, format each as:
  `<char>{character}</char> <moral>{moral}</moral>\n{story}</story>`
  then tokenize. Split train/val (e.g. 95/5) on a subset of **100k–500k** fables.
- **Why a subset:** 3M is overkill for 200M params; 100k–500k trains fast on a T4 and
  still teaches fable structure + morals. Configurable via `--n`.

---

## 3. Two Colab notebooks

### Notebook A — `notebooks/train_fable_colab.ipynb` (train → checkpoint)
1. **Setup**: `pip install torch ninja` (+ `tiktoken`/`tokenizers`); clone/inline
   nanoGPT (`model.py`, `train.py`). Run via `colab run --gpu T4`.
2. **Tokenizer**: train a BPE tokenizer on the subset; persist `tokenizer.json`.
3. **Data prep**: build the prefix-formatted `.bin`/`.npz` shards (nanoGPT format).
4. **Train**: `adamw` + cosine LR (`lr≈3e-4`, `batch≈64` w/ grad-accum, `max_iters`
   scaled to subset, `warmup`, `weight_decay`), `bf16` AMP. Save best `ckpt.pt` to
   `/content/drive/...` (mount via `colab drivemount`).
5. **Export**: keep `ckpt.pt`; optionally convert to GGUF (q8) for the local app.

### Notebook B — `notebooks/eval_gen_fable_colab.ipynb` (eval + generation)
1. **Load** `ckpt.pt` (+ tokenizer) on Colab GPU.
2. **Generate**: feed `<char>{x}</char> <moral>{y}</moral>` seeds → sample fables.
3. **Eval**: 4-axis LLM-as-judge (Grammar, Creativity, Moral Clarity, Prompt
   Adherence) reusing `docs/adr/0002-evaluation-methodology.md`. Compare **from-scratch
   200M vs a reference base** (e.g. `qwen3:4b` via Ollama-in-Colab or API) on held-out
   seed prompts; emit `results/eval_summary.json`.

---

## 4. `google-colab-cli` runbook

```bash
# --- Train (Notebook A) ---
colab new -s trainer --gpu T4
colab drivemount -s trainer
colab exec -s trainer -f notebooks/train_fable_colab.ipynb
colab download -s trainer /content/drive/fable200m-ckpt.pt ./models/
colab stop -s trainer

# --- Eval/Gen (Notebook B) ---
colab run --gpu T4 notebooks/eval_gen_fable_colab.ipynb
```
`colab run` auto-provisions, executes, and tears down; `--keep` + `colab download`
when you need the checkpoint.

---

## 5. Local integration

| Piece | Change |
|---|---|
| `config/models.json` | Add `{id:"fable-200m", kind:"finetuned", ...}`. If exported to GGUF → `ollama create`; else run the 200M via a local inference server (e.g. `llama.cpp` server) and point Ollama/HTTP at it. |
| `app/prompt_en.py` | Add a `build_seed_prompt(character, moral)` emitting the `<char>..</char> <moral>..</moral>` prefix the model was trained on. |
| Compare / Results | Already pit `base` vs `finetuned` from registry — works unchanged. |
| README §13 | Rewrite to the two-notebook Colab pipeline above. |

No RAG/retrieval: generation is pure sampling conditioned on the two keyword seeds.

---

## 6. Tasks (checklist)

- [ ] **T1** `scripts/prepare_tf1.py`: stream TF1, parse `character`+`moral`, format
      prefix samples, build nanoGPT `.bin` shards + train BPE tokenizer (TDD on fixture).
- [ ] **T2** `notebooks/train_fable_colab.ipynb`: setup → tokenizer → data → 200M
      nanoGPT train → Drive export (`ckpt.pt` + optional GGUF).
- [ ] **T3** `notebooks/eval_gen_fable_colab.ipynb`: load ckpt → gen → 4-axis judge →
      `results/eval_summary.json` (200M vs reference base).
- [ ] **T4** `google-colab-cli` runbook in README §13 + `docs/runbooks/colab-train.md`.
- [ ] **T5** Register `fable-200m` in `config/models.json`; wire `build_seed_prompt`;
      smoke in app (or via local inference server).
- [ ] **T6** `docs/adr/0003-...` recording the from-scratch-200M decision (or update ADR-0001).
- [ ] **T7** Correct README (fine-tune language → from-scratch 200M nanoGPT on Colab).

---

## 7. Open questions

1. **Subset size**: default 200k fables? (scales train time on T4)
2. **Export format**: keep `ckpt.pt` + run locally via `llama.cpp`/a tiny server, or
   convert to GGUF for direct Ollama use in the app?
3. **Reference for eval**: compare against `qwen3:4b` (local base) or skip the
   comparison and just report absolute 4-axis scores for the 200M model?
