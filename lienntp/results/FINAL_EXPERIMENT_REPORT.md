# Final Experiment Report

Date: 2026-07-18

## Project Scope

The project studies local English fable generation from structured prompts with five slots:

- Character
- Setting
- Challenge
- Outcome
- Teaching / moral

The fixed evaluation set is `data/test_prompts.jsonl` with 25 prompts. The main research question is:

> Can prompt control, post-processing, repair, or LoRA fine-tuning improve reliability and fluency for local fable generation?

## Compared Systems

| System | Type | Main role |
| --- | --- | --- |
| Base FP16 | Base instruct model | Fluency baseline |
| SFT Clean 3K | Broad SFT | Early fine-tuning contrast |
| Failure LoRA 300 | Targeted LoRA | Tests whether small failure-focused data fixes common errors |
| Strict Prompt | Prompt engineering | Tests whether stronger instructions improve control |
| Strict + Postprocess | Prompt + deterministic rule | Tests whether prompt fluency can be kept while fixing moral format |
| Base + Repair | Error-triggered rewrite + postprocess | Best reliability-oriented pipeline so far |
| Fluency SFT v1 LoRA | Quality-filtered 10K SFT | Tests whether larger filtered data improves fluency |

## Dataset Contributions

### Failure-Focused LoRA 300

Path:

```text
data/failure_focused_lora_300.zip
```

This dataset was built from observed failure types and corrected examples. It is intentionally small and targeted. It is not meant to teach the whole writing style; it tests whether a small LoRA can repair specific errors such as missing moral lines, poor endings, and weak prompt adherence.

Result: useful as a negative result. The LoRA did not learn reliable final moral behavior.

### Fluency SFT v1 10K

Paths:

```text
data/fluency_sft_v1/train.jsonl
data/fluency_sft_v1/valid.jsonl
data/fluency_sft_v1/dataset_report.md
```

Summary:

| Item | Value |
| --- | ---: |
| Total accepted | 10,000 |
| Train rows | 9,000 |
| Valid rows | 1,000 |
| Source | `klusai/ds-tf1-en-3m` |
| Average words | 264.3 |
| Scanned rows | 18,970 |
| Rejected rows | 8,970 |

Filters rejected samples with excessive length, long sentences, unsafe terms, duplicated/multiple morals, repetition, meta text, and problematic formatting. The goal was to improve fluency with a larger but cleaner SFT dataset.

Result: it did not improve fluency overall compared with Base+Repair.

## Automatic Evaluation

| Metric | Base | Strict Prompt | Strict + Postprocess | Base + Repair | SFT Clean 3K | Failure LoRA | Fluency SFT v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Total prompts | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| Success | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| Moral footer rate | 0.52 | 0.56 | 1.00 | 1.00 | 0.56 | 0.00 | 0.16 |
| Empty moral rate | 0.08 | 0.00 | 0.00 | 0.00 | 0.04 | 0.28 | 0.76 |
| Exact character rate | 0.12 | 0.92 | 0.92 | 0.12 | 0.04 | 0.20 | 0.28 |
| Exact moral rate | 0.16 | 0.56 | 1.00 | 0.76 | 0.52 | 0.00 | 0.16 |
| Outcome coverage | 0.84 | 0.80 | 0.80 | 0.84 | 0.68 | 0.76 | 0.84 |
| Clean ending | 0.76 | 0.96 | 1.00 | 1.00 | 0.20 | 0.72 | 0.12 |
| Run-on rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 0.04 | 0.00 |

Key automatic finding:

- Strict + Postprocess is the strongest automatic reliability setup.
- Base + Repair is also strong and improves moral reliability.
- Both LoRA directions underperform on moral footer reliability.
- Fluency SFT v1 has many blank `Moral:` endings despite being trained on filtered data.

## Human Evaluation

### Final 5-Way Evaluation

Source:

```text
results/human_eval_final_5way_10.md
```

| Model | English fluency | Prompt adherence | Fable structure | Moral clarity | Child safety | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base FP16 | 3.40 | 4.00 | 4.00 | 3.70 | 4.80 | 3.98 |
| SFT Clean 3K | 2.20 | 3.60 | 3.00 | 3.80 | 4.90 | 3.50 |
| Failure LoRA | 3.10 | 3.80 | 3.20 | 2.50 | 5.00 | 3.52 |
| Strict Prompt | 3.70 | 4.10 | 3.50 | 3.80 | 4.60 | 3.94 |
| Base + Repair | 3.40 | 4.30 | 4.10 | 5.00 | 4.80 | 4.32 |

Conclusion: Base + Repair is the best overall system in this evaluation.

### Fluency SFT v1 Evaluation

Source:

```text
results/human_eval_fluency_sft_v1_10.md
```

| Model | English fluency | Prompt adherence | Fable structure | Moral clarity | Child safety | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base + Repair | 3.40 | 4.30 | 4.10 | 5.00 | 4.80 | 4.32 |
| Strict + Postprocess | 3.70 | 4.10 | 3.60 | 5.00 | 4.60 | 4.20 |
| Fluency SFT v1 | 3.00 | 3.70 | 4.00 | 3.70 | 4.80 | 3.84 |

Fluency SFT v1 versus Base + Repair:

| Result | Prompts | Count |
| --- | --- | ---: |
| Better fluency than Base + Repair | p03, p10 | 2 |
| Equal fluency | p02, p06, p09 | 3 |
| Worse fluency | p01, p04, p05, p07, p08 | 5 |

Conclusion: Fluency SFT v1 is not more fluent overall. It improves some prompts, but regresses more often. The main errors are inconsistent characters, pronoun shifts, malformed openings, overlong or illogical sentences, incomplete endings, and blank `Moral:` lines.

## Per-Experiment Conclusions

### 1. Base Model

The base instruct model has strong general fluency and coherent story writing. Its weakness is reliability: it may omit the final moral, leave `Moral:` blank, or fail strict slot matching.

Conclusion: good writing baseline, but not reliable enough without control.

### 2. SFT Clean 3K

SFT Clean 3K is faster and sometimes more direct about moral wording, but it weakens fluency, structure, and endings.

Conclusion: broad SFT with weakly controlled synthetic data hurts narrative quality.

### 3. Failure-Focused LoRA 300

The 300-row targeted LoRA did not fix the intended moral footer behavior. Its final moral reliability is worse than Base + Repair and Strict + Postprocess.

Conclusion: small failure-focused LoRA is academically useful as a negative result. The dataset size and/or training objective were not enough.

### 4. Strict Prompt

Strict prompting improves exact character control and clean endings. However, prompt-only control still does not fully solve moral footer reliability.

Conclusion: prompt engineering helps, but is not sufficient alone.

### 5. Strict + Postprocess

This system keeps the strict prompt output and applies deterministic moral-line cleanup. It reaches perfect automatic moral footer and clean ending scores.

Conclusion: this is the best lightweight reliability intervention and has the best human fluency among the final comparison systems.

### 6. Base + Repair

This system uses postprocess and triggers repair only for severe detected problems. It is the best human-rated overall system.

Conclusion: best final candidate for the project because it balances story quality, adherence, structure, and moral clarity.

### 7. Fluency SFT v1 LoRA

This LoRA was trained on 10K quality-filtered fables. Despite the larger filtered dataset, it did not improve fluency over Base + Repair and produced many blank moral fields.

Conclusion: larger SFT data alone is still not enough. Fine-tuning may need better target formatting, shorter outputs, stronger assistant template checks, or a different objective.

## Research Contribution

This project has a valid practical research angle:

> For a strong local instruct model, reliability improvements from prompt control and deterministic/error-triggered correction can outperform small or moderately sized LoRA fine-tuning for structured children's fable generation.

The work contributes:

- A fixed 25-prompt benchmark for structured fable generation.
- Automatic metrics for moral footer reliability, exact moral match, slot coverage, clean ending, and run-on detection.
- Human evaluation across fluency, adherence, structure, moral clarity, and safety.
- Negative results for broad SFT, small failure-focused LoRA, and larger filtered fluency SFT.
- A practical final pipeline showing that postprocess/repair can outperform fine-tuning under limited compute and dataset quality constraints.

## App Integration Status

### Already Integrated

The app reads model options from:

```text
config/models.json
```

The new Fluency SFT v1 model has been added:

```text
fluency-sft-v1-lora-q4
```

It maps to the Ollama model:

```text
llama32-fable-fluency-sft-v1:q4
```

The model was imported into Ollama successfully. Because the frontend calls `/models`, this model should appear in the app model selector and compare mode after the backend is running.

Also integrated through the registry:

```text
sft-llama32-3b-clean3k
failure-lora-llama32-3b
fluency-sft-v1-lora-q4
```

### Generation Modes Integrated

The app now supports selectable generation modes through the live `/generate/stream` endpoint:

```text
raw
postprocess
repair
```

Mode behavior:

| Mode | Behavior |
| --- | --- |
| Raw | Direct Ollama generation from the selected model |
| Postprocess | Direct generation plus deterministic final `Moral:` normalization |
| Repair | Validate the draft, rewrite severe failures with the configured repair model, then normalize the final moral |

The frontend exposes these modes as:

```text
Raw
Post
Repair
```

The default repair model id is:

```text
sft-llama32-3b-clean3k
```

It can be changed with:

```text
FABLE_REPAIR_MODEL_ID
```

## Recommended Final Claim

Use `Base + Repair` as the main final system in the report, and mention `Strict + Postprocess` as the strongest lightweight alternative.

Do not promote Fluency SFT v1 as the final model. It is useful as a research result showing that even a 10K filtered SFT dataset can fail to improve fluency/reliability if the target distribution or output formatting is not strong enough.

## Next Implementation Step

For demonstration, use:

| Demonstration target | App selection |
| --- | --- |
| Raw fine-tuned model behavior | Select the LoRA model and `Raw` mode |
| Lightweight reliability intervention | Select a base model and `Post` mode |
| Best reliability-oriented pipeline | Select a base model and `Repair` mode |

This makes the main experimental contribution visible inside the web app.
