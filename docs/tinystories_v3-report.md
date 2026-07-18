> **Note:** This is the report for the companion project **tinystories_v3**, ported into `tinystory-vn` (docs only). All **code, ADRs, tests, and notebooks** referenced below live in the source repo: https://github.com/harryct229/tinystories_v3 — relative paths like `src/`, `docs/adr/`, `tests/`, `notebooks/` refer to that repo. An HTML version sits beside this file: `tinystories_v3-report.html`.

---

# Where Should the Adapters Go? A LoRA Adapter-Placement Study on SmolLM2-135M for Moral-Fable Generation

**Project:** `tinystories_v3` · **Course:** Generative AI · **Type:** Technical report
**Base model:** SmolLM2-135M · **Dataset:** TF1-EN-3M (`klusai/ds-tf1-en-3m`) · **Compute:** 1× Colab L4 · **Date:** 2026-07-13
**Code:** `github.com/harryct229/tinystories_v3` · **Adapters:** `hf.co/congthanh991/tsv3-smollm135-{A-qv-all,B-qv-last3,C-alllinear}`

---

## Abstract

The *TinyStories* line of work established that language models far below a billion parameters can produce
coherent children's narratives when trained on narrow, high-quality synthetic text. Building on that premise, we
take a pretrained **SmolLM2-135M** and fine-tune it with **Low-Rank Adaptation (LoRA)** on **TF1-EN-3M**, a corpus
of three million synthetic English moral fables, and make a single question the object of study: **where the
low-rank adapters should be placed.** Holding rank, data, and optimisation schedule fixed, we compare an untrained
floor against three placements — attention `q,v` across all layers (**A**), the same restricted to the last third
of layers (**B**), and all seven linear projections across all layers (**C**). On a fixed held-out sample,
validation perplexity falls from **9.52** (base) to **3.84** (C), and Flesch Reading Ease moves from **−66**
(effectively unreadable) to **+53** (age-appropriate). Two clean one-factor contrasts answer the title: adapting
*all* layers beats the last third (4.82 vs. 5.46), but the dominant gain by far comes from **breadth** — adding the
MLP projections (3.84 vs. 4.82). The project is framed as a deliberate counterpoint to a sibling project that
full-fine-tunes a 4-billion-parameter model, trading absolute quality for a controlled look at parameter-efficient
placement, and it is delivered end-to-end: trained, evaluated, exported to GGUF, and served through Ollama inside
the sibling project's application.

---

## 1 · Introduction

### 1.1 Motivation

Fables are a compact, highly structured narrative form: a character with a trait, a setting, a challenge, a
resolution, and an explicit moral. That regularity makes them an unusually good testbed for small language models —
the vocabulary is limited, the arc is short, and quality is legible to any reader. Eldan & Li's *TinyStories* [2]
demonstrated that models with fewer than 10M parameters can write coherent children's stories when the training
data is curated and narrow, and the dataset we use [1] was constructed explicitly to be "conducive to
parameter-efficient fine-tuning of small downstream models."

Rather than train from scratch, we start from a **pretrained** small model and adapt it with **LoRA** [3], which
freezes the base weights and learns a small number of low-rank update matrices inserted into selected projections.
LoRA is normally deployed with placement chosen by convention ("attach adapters to `q` and `v`"). The brief for
this project made that convention the contribution: *"note why we select which layers to add to our model."* We
read "which layers to add" precisely as **adapter placement**, and built the entire study around it.

### 1.2 Research question

> Under a fixed budget — the same base model, LoRA rank, training data, and schedule — **does adapter placement
> measurably change fable quality, and which axis matters more: the *layers* the adapters span, or the *modules*
> they attach to?**

### 1.3 Contributions

1. A controlled, single-variable **ablation of LoRA placement** for narrative generation on a 135M model, isolating
   the *layer-depth* axis (A vs. B) and the *module-breadth* axis (A vs. C).
2. A quantitative result: **module breadth dominates layer depth** for this task; adding MLP adapters yields the
   largest single improvement.
3. A complete, reproducible pipeline — data, training, evaluation, and a **GGUF → Ollama** export that plugs the
   result into a sibling application for a live small-vs-large comparison.

---

## 2 · Background & theory

### 2.1 Low-Rank Adaptation (LoRA)

A dense layer computes `h = W·x` with a weight matrix `W ∈ ℝ^{d_out × d_in}`. Full fine-tuning updates all
`d_out · d_in` entries of `W`. LoRA instead freezes `W` and learns a **low-rank** update

```
    W' = W + ΔW ,   ΔW = (α / r) · B · A ,   A ∈ ℝ^{r × d_in},  B ∈ ℝ^{d_out × r}
```

with rank `r ≪ min(d_in, d_out)`, `B` initialised to zero (so training starts exactly at the pretrained model), and
a fixed scaling `α / r`. Only `A` and `B` are trained, giving `r·(d_in + d_out)` parameters per adapted matrix —
orders of magnitude fewer than `d_in·d_out`. At inference the product `BA` can be merged back into `W`, so a
LoRA-adapted model has **zero additional latency** once merged (we rely on exactly this to export a single dense
model to GGUF, §7).

### 2.2 Why placement is a real question

LoRA leaves two design choices unspecified, and they are the two axes of this study:

- **Which modules?** A transformer block contains attention projections (`q, k, v, o`) and MLP projections
  (`gate, up, down`). The original LoRA paper reports that adapting only `W_q` and `W_v` is often sufficient, which
  became the community default — but "often sufficient" is a claim to be tested per task, not a law.
- **Which layers?** Adapters can be attached to every layer or only a subset. Later layers are frequently argued to
  carry more task-specific, higher-level representation, which motivates a "last-third-only" placement as an
  efficiency lever.

These two axes are usually entangled in practice. The experimental design in §6.3 **deliberately separates them**
so each can be read on its own.

### 2.3 Why the fable form suits a small model

The target style is deliberately narrow: ~200-word children's fables with a fixed five-element structure and a
limited vocabulary. A 135M model has neither the capacity nor the need to model open-domain text; the task's
narrowness is what makes a tiny model viable, and what makes the placement question crisp — there is a real signal
to capture, and small differences in *where* capacity is added are visible in the metrics.

---

## 3 · The dataset

We train on **TF1-EN-3M** (`klusai/ds-tf1-en-3m`) [1], three million synthetic English moral fables generated by an
8-billion-parameter instruction model. Each row pairs a **structured prompt** with a **fable** and a fixed **system
message**. The prompt renders five labelled slots the model must weave into a story:

| Slot | Meaning | Example |
|---|---|---|
| **Main Character** | protagonist (character + trait folded together) | *a clever skunk* |
| **Setting** | where the story unfolds | *a flower field* |
| **Challenge** | the central conflict | *rivalry in love* |
| **Outcome** | how it resolves | *ancient enemies sign a pact* |
| **Teaching** | the moral | *appearances can be deceiving* |

Splits are **2.8M train / 100K validation / 100K test**. We reuse the paper's dataset and evaluation vocabulary
rather than reinventing criteria. A representative training pair — shown to fix the *target style*, not as a model
output — reads:

> **Prompt** — Main Character: *a clever skunk* · Setting: *a flower field* · Challenge: *rivalry in love* ·
> Outcome: *ancient enemies sign a pact* · Teaching: *appearances can be deceiving*.
> **Fable** — "In a sun-kissed flower field, a clever skunk loved to sniff out the sweetest blooms … As they bent to
> drink, they saw their reflections in the calm water. 'We've been judging each other wrong,' said the skunk … From
> that day on they promised to look beyond appearances — a reminder that true beauty comes from within."

---

## 4 · Positioning: how this differs from `tinystory-vn`

This project is a deliberate counterpoint to a sibling course project, `tinystory-vn`, which targets the same
dataset and output style from the opposite end of the design space. That project **full-fine-tunes a 4-billion-
parameter model** (Qwen3-4B, SFT + ORPO) and centres its contribution on the **application and evaluation
methodology** — a served app with a multi-layer guardrail and an LLM-judge panel. We deliberately go small and make
the **model-internal placement question** the object of study, an axis that project never explores. The two are
complementary; §8.6 lays the angles side by side.

---

## 5 · Method

### 5.1 Base model — and why SmolLM2-135M

We start from **SmolLM2-135M** (base, non-instruct) [4]: a Llama-style decoder with **30 layers**, hidden size
**576**, **9 attention heads** (3 KV heads, grouped-query attention), MLP intermediate size 1536, and a 49,152-token
vocabulary. The choice is load-bearing for the study:

- **Separable projections.** SmolLM2 keeps distinct `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj` matrices. This is what makes the textbook "adapt `q` and `v` only" contrast *well-defined*. A model
  with a **fused QKV** projection (e.g. GPT-2's `c_attn`, which packs `q,k,v` into one matrix) cannot separate `q`
  from `v`, so the module-axis contrast would be impossible. (See ADR-0001.)
- **Base, not instruct.** Starting from a base checkpoint gives an honest before/after: the base model cannot follow
  the fable prompt at all, so the fine-tuning effect and the placement differences are both visible against a real
  floor.

### 5.2 Task formulation

We frame learning as **conditional generation**. The model input is the concatenation `system_message ⧺ prompt`;
the target is the `fable`. Crucially the cross-entropy loss is computed on the **fable tokens only** — every context
(system + prompt) token is masked to the ignore index `−100` (completion-only masking) — and an end-of-text token is
appended so the model learns to stop:

```
input  =  BOS  s_1 … s_m  (system+prompt)   f_1 … f_n  <eot>
labels = −100 −100 … −100  (masked context)  f_1 … f_n  <eot>
loss   =  −(1/N) · Σ_{t: label_t ≠ −100}  log p(label_t | x_{<t})
```

This keeps the model controllable at inference (the story is steered by the five slots) and, critically for a
placement study, makes **perplexity a fair, like-for-like signal** — every arm is scored under identical masking on
identical text.

### 5.3 The ablation — which layers, which modules

Rank is fixed at **`r = 16`** (`α = 32`, dropout 0.05) across all arms so that **placement is the only variable**.
Four configurations are trained and evaluated:

| Arm | Adapters on | Layers | Isolates |
|---|---|---|---|
| **base** | — (no fine-tuning) | — | reference floor |
| **A** | `q_proj, v_proj` | all 30 | — |
| **B** | `q_proj, v_proj` | last 10 (indices 20–29) | *layer depth* (vs. A) |
| **C** | all 7 linear projections | all 30 | *module breadth* (vs. A) |

- **A vs. B** holds the module set constant (`q,v`) and varies only **layer depth**.
- **A vs. C** holds layer coverage constant (all 30) and varies only **module breadth**.

Two clean, single-factor comparisons — the entire reason the arms are shaped this way. (See ADR-0002.) We verify
programmatically that arm B's adapters really are confined to layers 20–29 and arm A's cover all 30, so the
"which-layers" claim is not merely nominal (`tests/test_arms.py`).

### 5.4 Adapted-parameter accounting

For a matrix of shape `(d_in, d_out)`, LoRA at rank `r` adds `r·(d_in + d_out)` parameters. With SmolLM2's dims:

| Projection | shape (in→out) | params @ r=16 |
|---|---|---|
| `q_proj` | 576 → 576 | 18,432 |
| `k_proj` | 576 → 192 | 12,288 |
| `v_proj` | 576 → 192 | 12,288 |
| `o_proj` | 576 → 576 | 18,432 |
| `gate_proj` | 576 → 1536 | 33,792 |
| `up_proj` | 576 → 1536 | 33,792 |
| `down_proj` | 1536 → 576 | 33,792 |

Multiplying by the adapted layers gives each arm's budget (against a ≈134.5M base):

| Arm | per-layer | × layers | total | % of model |
|---|---|---|---|---|
| **A** (`q,v`, 30) | 30,720 | 30 | **≈ 0.92M** | ≈ 0.68% |
| **B** (`q,v`, 10) | 30,720 | 10 | **≈ 0.31M** | ≈ 0.23% |
| **C** (all-7, 30) | 162,816 | 30 | **≈ 4.88M** | ≈ 3.5% |

### 5.5 Training configuration

Each arm trains on a **fixed 50,000-fable subset** (seed 42, identical across arms) for **2 epochs** at sequence
length 512, using AdamW, learning rate `2e-4`, a cosine schedule with 3% warmup, bf16 precision, per-device batch 16
with gradient accumulation 2 (effective batch 32, ≈ **3,125 steps/arm**). Training ran on a single **Colab L4**;
each arm's adapter (a few MB) is pushed to the Hugging Face Hub, and metrics stream to Weights & Biases with a
heartbeat callback for liveness.

### 5.6 Evaluation methodology

On a **fixed 500-row held-out sample** of the validation split (seed 42) we report:

- **Validation perplexity** (primary): teacher-forced, loss on fable tokens only (same masking as training), so the
  base and every arm are scored identically. Perplexity is `exp` of the mean per-token cross-entropy, accumulated
  weighted by token count (not a naïve mean of per-row means). It is the natural, API-free, directly-comparable
  signal for a placement study.
- **Reference-free text metrics**, computed on 100 sampled generations per arm (temperature 0.8, top-p 0.9,
  repetition penalty 1.3, seeded for reproducibility):
  - **Distinct-1 / Distinct-2** — ratio of unique unigrams / bigrams to total (lexical diversity).
  - **Self-BLEU** — mean BLEU of each generation against the rest (intra-set redundancy; lower = more diverse).
  - **Flesch Reading Ease** — a readability score from sentence and syllable counts; higher is easier, and values
    below zero indicate text that is effectively unreadable.
- **LLM-as-judge (planned)** — a single local judge over the paper's four axes (grammar, creativity, moral clarity,
  prompt adherence). Deferred as a stretch goal; the quantitative metrics already carry the finding. (See ADR-0003.)

---

## 6 · Results

All four configurations, evaluated identically on the same 500 held-out prompts:

| Config | Adapted params | **Val PPL ↓** | Distinct-1 | Distinct-2 | Self-BLEU | Flesch |
|---|---:|---:|---:|---:|---:|---:|
| **base** (no FT) | 0 | 9.52 | 0.557 | 0.971 | 0.007 | −66.2 |
| **A** — `q,v` · all-30 | ≈ 0.9M | 4.82 | 0.188 | 0.716 | 0.176 | **57.7** |
| **B** — `q,v` · last-10 | ≈ 0.3M | 5.46 | 0.190 | 0.739 | 0.171 | 51.1 |
| **C** — all-linear · all-30 | ≈ 4.9M | **3.84** | **0.210** | 0.728 | 0.191 | 52.8 |

Ranking by perplexity: **C (3.84) < A (4.82) < B (5.46) ≪ base (9.52).**

Headline numbers:

- **−60%** perplexity vs. the untrained floor, for the best config (C).
- **3.5%** of the model's weights trained to reach that result.
- **+119 Flesch points** (−66 → +53) — from unreadable to age-appropriate.

---

## 7 · Analysis & discussion

### 7.1 Fine-tuning works — dramatically

The untrained base cannot follow the fable prompt: perplexity 9.52 and a strongly negative Flesch score signal text
that is not readable children's prose. Its **high** Distinct-1 (0.557) and **near-zero** Self-BLEU (0.007) are *not*
a quality signal — they are the fingerprint of **random, unconstrained** text, which trivially avoids repetition.
Every LoRA arm collapses that randomness into coherent, templated fables (Self-BLEU ≈ 0.17–0.19, Flesch +51…+58).
This is a caution about reference-free diversity metrics in isolation: without a competence floor, "more diverse"
can simply mean "less coherent."

### 7.2 Which layers? (A vs. B)

**Adapting all layers beats the last third — PPL 4.82 vs. 5.46.** Layer coverage helps. But B recovers roughly
**85%** of A's perplexity improvement over the base while adapting only **one third** as many layers (≈ 0.3M vs.
≈ 0.9M parameters). Restricting adapters to later layers is therefore a **defensible efficiency trade** — not a free
lunch, but a favourable one when parameter or compute budget is the binding constraint.

### 7.3 Which modules? (A vs. C)

**All-linear crushes attention-only — PPL 3.84 vs. 4.82, a ≈ 20% reduction.** The single largest gain in the whole
study comes from adding adapters to the **MLP** projections (`gate, up, down`), not from covering more layers. For
fable generation on this model, module **breadth** matters more than layer **depth**. A plausible reading: the MLP
sublayers carry much of a transformer's capacity to store and recombine the surface-level, lexical patterns that a
narrow generation task like fable-writing leans on, so giving them trainable capacity pays off disproportionately.

### 7.4 The best configuration, and a trade-off

**C (all-linear, all layers)** is the best configuration at PPL 3.84 and also the richest vocabulary among the
trained arms (Distinct-1 0.210); it is exported as `tsv3-smollm135-best`. One subtlety is worth naming: C wins
decisively on modelling, while **A** scores highest on Flesch readability (57.7 vs. 52.8) — the all-linear model
writes slightly denser prose. For the pedagogical target, C's stronger modelling is the headline; A is a lightweight
runner-up worth keeping in mind when compute is scarce.

### 7.5 Qualitative check

Driving the exported models on an unseen prompt (*a brave little turtle · a quiet pond · a sudden storm · the animals
work together · teamwork overcomes fear*) shows the difference concretely. The **base** model goes off-task, emitting
essay-writing meta-commentary ("What type of writing will you create? … Who's who: which character has most
authority?"). The **best (C)** model writes a real fable:

> *"In the quiet pond, where water lilies swayed gently in the breeze and fish swam happily by, a brave little turtle
> lived among his friends… One day, dark clouds gathered over the pond as strong winds howled and loud thunderclaps
> rumbled… The clever rabbit, strong and swift, had been watching from a nearby rock. He suggested that they work
> together to save their friends… The two groups of creatures huddled closer and began working together…"*

Minor artifacts (an occasional "(Figure 1)", slight character drift) are expected from a Q8-quantised 135M model and
the prompt-format nuance discussed in §9, but the output is unmistakably a coherent, on-prompt fable.

### 7.6 Cross-project comparison

| Angle | `tinystories_v3` (this) | `tinystory-vn` |
|---|---|---|
| Base size | 135M | 4B (≈ 30× larger) |
| Adaptation | LoRA, adapter-placement study | full SFT + ORPO |
| Research question | which layers / modules to adapt | app + evaluation methodology |
| Compute | 1× L4, minutes per arm | substantially larger |
| Controllability | 5-slot conditional prompt | 5-slot conditional prompt |
| Delivery | GGUF adapters imported into their app | full FastAPI + React app + guardrail + judge |

The intended demo is literal: the best adapter (and the base) are merged, converted to GGUF, and registered in
`tinystory-vn`'s Ollama model list, so its existing **Compare** mode can put our 135M fine-tune head-to-head against
their 4B model — a live "30× smaller, how close?" comparison. (See ADR-0004.)

---

## 8 · Deployment: GGUF → Ollama → tinystory-vn

The academic result is delivered as a runnable model. For **base** and **best (C)**:

1. **Merge** — pull the base and the C adapter from the Hub and `merge_and_unload()` into a single dense model.
2. **Convert** — `llama.cpp`'s `convert_hf_to_gguf.py` writes a **GGUF Q8_0** (≈ 138 MB each). SmolLM2 is Llama-arch,
   so conversion is direct.
3. **Modelfile** — an Ollama `Modelfile` whose `TEMPLATE` reproduces the training format (`{{ .System }}` ⧺ blank
   line ⧺ `{{ .Prompt }}` ⧺ blank line) and stops on `<|endoftext|>`, with the TF1 system message as default.
4. **`ollama create`** — `tsv3-smollm135-base` and `tsv3-smollm135-best`.
5. **Register** — append two entries to `tinystory-vn`'s `config/models.json` (`kind: base` and `kind: finetuned`).

Both Ollama models were created and verified **behaviourally distinct** (base off-task; best writes fables, §7.5).

---

## 9 · Limitations & threats to validity

- **Absolute ceiling.** A 135M model caps quality well below a multi-billion model; the study is about *relative*
  placement efficiency, not beating a large model.
- **Single run, no intervals.** Each arm is one seed and one schedule; we report point estimates without confidence
  intervals. The perplexity gaps (3.84 / 4.82 / 5.46) are comfortably separated, but a multi-seed repeat would harden
  them.
- **Incomplete 2×2.** We run three of the four cells; the missing `all-linear × last-third` cell would let us test
  for an interaction between the two axes rather than reading them independently.
- **No judge panel yet.** The four-axis LLM-as-judge (and inter-judge agreement) is deferred; conclusions rest on
  perplexity and reference-free metrics.
- **Prompt-format mismatch downstream.** The adapters are trained on TF1's exact wording; when driven by
  `tinystory-vn`'s slightly different prompt, a brittle 135M model may score lower than in our own eval. Accepted
  deliberately (ADR-0004).
- **Light data shuffle.** The 50k subset is drawn through a streaming shuffle with a 10k buffer — deterministic and
  identical across arms (so the comparison stays fair), but closer to "lightly shuffled head" than a uniform sample.

---

## 10 · Reproducibility

- **Code:** `github.com/harryct229/tinystories_v3` — logic is unit-tested on CPU; training/eval carry tiny-model
  smoke tests. Run `python -m pytest`.
- **Adapters:** public on the Hub — `congthanh991/tsv3-smollm135-{A-qv-all, B-qv-last3, C-alllinear}`.
- **Training / eval:** `notebooks/colab_runner.ipynb` (or `python -m src.train --arm {A,B,C} --push`), then the eval
  cells; results in `results_auto.json`.
- **Two field notes for a Colab rerun** (recorded so they don't recur): (i) Colab preinstalls `torchao 0.10.0`, which
  makes PEFT's LoRA injection raise on a version check — `pip uninstall -y torchao` (we don't use it); (ii) the Colab
  CLI kernel socket is unreliable for long calls — run training as a **background job** and observe progress through
  the Hub (adapters appearing), not by tailing logs.

---

## 11 · Conclusion & future work

Under a strictly controlled budget, adapter **placement measurably changes** how well a small model learns to write
fables. The two contrasts give a crisp, reportable answer: covering *all* layers beats the last third, but the
dominant lever is **module breadth** — attaching adapters to the MLP projections, not just attention, drives the best
result (PPL 3.84, a 60% reduction from the untrained floor) while training only 3.5% of the model. A tiny, pretrained
model plus well-placed low-rank adapters is enough to turn unreadable output into age-appropriate fables.

**Future work:** run the four-axis judge to corroborate the perplexity ranking; sweep the rank to separate placement
effects from added capacity; add the missing `all-linear × last-third` cell to complete the 2×2 and test for
interaction; and use the exported models for a quantitative small-vs-large comparison against the 4B sibling.

---

## References

1. Nadás, Dioșan, Piscoran, Tomescu (2025). *TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small,
   Open Language Models.* arXiv:2504.20605. Dataset: `klusai/ds-tf1-en-3m`.
2. Eldan & Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
3. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
   arXiv:2106.09685.
4. Allal et al. (2025). *SmolLM2.* `HuggingFaceTB/SmolLM2-135M`, Hugging Face Hub.
5. `tinystory-vn` — sibling course project (Qwen3-4B, SFT + ORPO). `github.com/tungd/tinystory-vn`.

---

## Appendix A · Exact configuration

```
base model       HuggingFaceTB/SmolLM2-135M   # Llama-style, 30 layers, hidden 576, 9/3 heads, vocab 49152
LoRA (fixed)     r=16  alpha=32  dropout=0.05  bias=none  task=CAUSAL_LM
  arm A          target=[q_proj,v_proj]                     layers=all 30
  arm B          target=[q_proj,v_proj]                     layers=20..29 (last third)
  arm C          target=[q,k,v,o,gate,up,down]_proj         layers=all 30
data / task      klusai/ds-tf1-en-3m  train subset=50,000 (seed 42)  epochs=2
                 conditional (system+prompt)->fable, loss on fable tokens only (-100 mask), max_seq_len=512
optimisation     AdamW  lr=2e-4  scheduler=cosine  warmup_ratio=0.03  bf16
                 per_device_batch=16  grad_accum=2  (effective 32)  ~3,125 steps/arm  hardware=Colab L4
evaluation       held-out=500 rows (validation, seed 42)
                 primary=perplexity (teacher-forced, fable tokens, token-weighted)
                 reference-free=Distinct-1/2, Self-BLEU, Flesch  (100 gens/arm; temp 0.8, top-p 0.9, rep-pen 1.3)
export           merge -> GGUF Q8_0 (llama.cpp) -> Ollama Modelfile -> tinystory-vn config/models.json
```

## Appendix B · Architecture Decision Records

| ADR | Decision |
|---|---|
| **0001** | Small pretrained model + LoRA (not from-scratch, not a big-model full fine-tune). |
| **0002** | The LoRA adapter-placement ablation *is* the contribution (arms A/B/C + base). |
| **0003** | Simplified single-judge evaluation instead of the paper's 3-judge panel. |
| **0004** | Deliver via `tinystory-vn` (GGUF/Ollama), no standalone app. |
