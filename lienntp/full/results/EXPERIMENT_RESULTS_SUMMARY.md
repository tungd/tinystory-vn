# Experiment Results Summary

Date: 2026-07-17

## Scope

This project evaluates English children's-fable generation from five structured
input slots:

- character
- setting
- challenge
- outcome
- teaching / moral

The fixed benchmark is `data/test_prompts.jsonl`.

## Compared Models

| Model | ID | Role |
| --- | --- | --- |
| Llama 3.2 3B Instruct FP16 | `base-llama32-3b-instruct` | Base model, no fine-tuning |
| Llama 3.2 3B Instruct Q4 + Strict Prompt | `base-llama32-3b-strict-prompt` | Prompt-control ablation |
| Llama 3.2 3B Instruct Q4 + Strict Prompt + Postprocess | `base-llama32-3b-strict-postprocess` | Fluency-preserving reliability candidate |
| Llama 3.2 3B Instruct + Postprocess | `base-llama32-3b-postprocess` | Rule-based reliability layer over base outputs |
| Llama 3.2 3B Instruct + Repair | `base-llama32-3b-repair` | Error-triggered rewrite plus postprocess |
| Llama 3.2 3B SFT Clean 3K Q4 | `sft-llama32-3b-clean3k` | Fine-tuned contrast model |
| Llama 3.2 3B Failure-Focused LoRA Q4 | `failure-lora-llama32-3b` | Failure-focused LoRA experiment |

## Artifacts

| Artifact | Path |
| --- | --- |
| Base generations | `results/baseline_outputs.jsonl` |
| Strict Prompt generations | `results/strict_prompt_outputs.jsonl` |
| Strict Prompt metrics | `results/eval_strict_prompt.md` |
| Strict Prompt + Postprocess generations | `results/strict_postprocess_outputs.jsonl` |
| Strict Prompt + Postprocess metrics | `results/eval_strict_postprocess.md` |
| SFT Clean 3K generations | `results/sft_clean3k_outputs.jsonl` |
| Base automatic metrics | `results/eval_base.md` |
| Base + Postprocess generations | `results/base_postprocess_outputs.jsonl` |
| Base + Postprocess metrics | `results/eval_base_postprocess.md` |
| Base + Repair generations | `results/base_repair_outputs.jsonl` |
| Base + Repair metrics | `results/eval_base_repair.md` |
| SFT automatic metrics | `results/eval_sft_clean3k.md` |
| Failure LoRA generations | `results/failure_lora_outputs.jsonl` |
| Failure LoRA metrics | `results/eval_failure_lora.md` |
| Human evaluation file | `results/human_eval_base_vs_clean3k_10.md` |
| 3-way human evaluation template | `results/human_eval_base_postprocess_sft10.md` |
| Final 4-way human evaluation template | `results/human_eval_final_4way_10.md` |
| Final 5-way human evaluation template | `results/human_eval_final_5way_10.md` |
| Detailed findings | `results/base_vs_sft_clean3k_findings.md` |
| Failure-focused LoRA dataset | `data/failure_focused_lora_300.zip` |

## Automatic Evaluation

| Metric | Base | Strict Prompt | Strict + Postprocess | Base + Postprocess | Base + Repair | SFT Clean 3K | Failure LoRA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Total prompts | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| Success | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| Moral label rate | 0.60 | 0.56 | 1.00 | 1.00 | 1.00 | 0.60 | 0.28 |
| Moral footer rate | 0.52 | 0.56 | 1.00 | 1.00 | 1.00 | 0.56 | 0.00 |
| Empty moral rate | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.28 |
| Exact character phrase rate | 0.12 | 0.92 | 0.92 | 0.12 | 0.12 | 0.04 | 0.20 |
| Exact moral keyword rate | 0.16 | 0.56 | 1.00 | 0.64 | 0.76 | 0.52 | 0.00 |
| Outcome coverage rate | 0.84 | 0.80 | 0.80 | 0.84 | 0.84 | 0.68 | 0.76 |
| Clean ending rate | 0.76 | 0.96 | 1.00 | 1.00 | 1.00 | 0.20 | 0.72 |
| Run-on sentence rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 0.04 |
| Avg words | 216.8 | 226.3 | 229.5 | 220.3 | 216.8 | 206.8 | 235.0 |
| Avg latency ms | 77348 | 21156 | 21156 | 77348 | 81575 | 16308 | 16460 |

## Human Evaluation

Manual scores were entered directly in
`results/human_eval_base_vs_clean3k_10.md`. Prompts p01-p09 are scored for both
models; p10 is still unscored.

| Criterion | Base | SFT Clean 3K |
| --- | ---: | ---: |
| English fluency | 4.11 | 2.44 |
| Prompt adherence | 4.11 | 3.44 |
| Fable structure | 3.78 | 2.89 |
| Moral clarity | 3.33 | 2.89 |
| Child safety | 5.00 | 5.00 |
| Average | 4.07 | 3.33 |

The final 5-way human evaluation was entered in
`results/human_eval_final_5way_10.md` and covers 10 prompts for each system.

| Model | English fluency | Prompt adherence | Fable structure | Moral clarity | Child safety | Overall average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base FP16 | 3.40 | 4.00 | 4.00 | 3.70 | 4.80 | 3.98 |
| SFT Clean 3K | 2.20 | 3.60 | 3.00 | 3.80 | 4.90 | 3.50 |
| Failure LoRA | 3.10 | 3.80 | 3.20 | 2.50 | 5.00 | 3.52 |
| Strict Prompt | 3.70 | 4.10 | 3.50 | 3.80 | 4.60 | 3.94 |
| Base + Repair | 3.40 | 4.30 | 4.10 | 5.00 | 4.80 | 4.32 |

Human ranking:

1. `Base + Repair`: 4.32
2. `Base FP16`: 3.98
3. `Strict Prompt`: 3.94
4. `Failure LoRA`: 3.52
5. `SFT Clean 3K`: 3.50

## Conclusion

The base Instruct model is stronger overall. It produces more fluent, coherent,
and complete fables. Its main weakness is formatting reliability: several
generations end with an empty `Moral:` or lack an explicit moral.

SFT Clean 3K improves some direct moral-keyword behavior and runs much faster in
Q4 form, but it significantly regresses in fluency, story structure, outcome
coverage, and clean endings. It should not be treated as the final champion.

The useful experimental finding is a trade-off:

- SFT can push the model toward explicit moral wording.
- Low-quality or weakly filtered synthetic SFT can damage narrative coherence.
- For this task, data quality and output-control reliability matter more than
  simply increasing the number of fine-tuning examples.

Final conclusion after the 5-way human evaluation: `Base + Repair` is the best
current system. It does not improve base fluency, but it preserves most of the
base model's story quality while substantially improving prompt adherence,
fable structure, and especially moral clarity. `Strict Prompt` is competitive
with the base and improves character control, but it is not enough to solve the
moral-footer reliability problem. Both fine-tuned models are useful negative
results: broad SFT hurts fluency, and the current failure-focused LoRA fails to
learn reliable explicit morals.

The first reliability intervention is promising: `Base + Postprocess` preserves
the base story body but fixes missing/empty moral lines and punctuation. It
raises moral footer rate, empty moral rate, and clean ending rate to perfect
scores on the 25-prompt benchmark without introducing run-on sentences.

The second intervention, `Base + Repair`, rewrites only outputs that fail severe
checks, using `SFT Clean 3K` as the available local rewrite model. It improves
exact moral keyword rate from 0.64 to 0.76 while keeping success, moral footer,
clean ending, and run-on sentence rates at their best values. Its latency is
slightly higher than postprocess because six prompts triggered rewrite.

This creates a useful ablation:

- Base: strong fluency, weak moral reliability.
- SFT Clean 3K: fast and moral-aware, but weaker prose.
- Strict Prompt: greatly improves exact character usage and clean endings, but
  still does not reliably produce the final `Moral:` footer.
- Strict Prompt + Postprocess: combines the best automatic controllability
  result so far with low latency. It should be the next candidate for human
  fluency evaluation.
- Base + Postprocess: fixes deterministic format failures.
- Base + Repair: adds targeted correction for severe structural failures.
- Failure LoRA: fast, but this checkpoint regresses heavily on the final
  `Moral:` requirement and should not be promoted.

The Failure LoRA result is still useful academically as a negative result:
targeted LoRA with the current 300-row mixture did not learn the desired moral
footer behavior. This suggests the failure-focused dataset construction or
training recipe needs revision before another LoRA run.

The Strict Prompt ablation answers the main prompt-engineering question:
stronger prompting can improve some controllability dimensions, especially
character inclusion, but prompt-only control is not sufficient for moral-footer
reliability. Error-driven repair remains the best automatic intervention in the
current experiment.

After cleanup, the most promising next fluency-focused direction is
`Strict Prompt + Postprocess`. It keeps the strict prompt's stronger character
control and likely fluency advantage, while postprocess fixes the explicit moral
line deterministically.

## Next Direction

To avoid duplicating the two reference projects:

- Do not follow the from-scratch SLM route.
- Do not follow DPO/ORPO preference alignment.

The recommended next direction is evaluation-driven correction for a strong
Instruct base model:

1. Keep the base model as the fluency baseline.
2. Keep `Base + Postprocess` as the minimal reliability intervention.
3. Add a lightweight rewrite/repair pass only when checks detect semantic or
   structural failures.
4. Evaluate `Base` vs `Base + Postprocess` vs `Base + Repair` vs `SFT Clean 3K`
   on the same 25 prompts.
5. If one more training run is required, train a small LoRA only on
   failure-focused examples targeting moral footer, clean ending, and slot
   adherence, rather than broad synthetic story generation.

This gives the project a distinct angle: practical reliability improvement for
local fable generation through measured error analysis and targeted correction.

## Failure-Focused LoRA Dataset

The training dataset for the next experiment has been prepared:

| Item | Value |
| --- | ---: |
| Total rows | 300 |
| Train rows | 270 |
| Validation rows | 30 |
| Observed Base-failure seed rows | 24 |
| TF1 augmented corrected-target rows | 276 |

Path:

```text
data/failure_focused_lora_300.zip
```

The dataset combines real observed Base failures with additional TF1 corrected
targets tagged by the same failure taxonomy. This gives the LoRA experiment a
clear research role: testing whether targeted failure-focused supervision can
improve reliability more efficiently than broad SFT.
