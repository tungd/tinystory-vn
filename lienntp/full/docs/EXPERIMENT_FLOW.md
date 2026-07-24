# Experiment Flow

This project now follows a fixed evaluation-first workflow. The goal is not to
train repeatedly until a model happens to look good, but to compare different
training approaches on the same task and report their trade-offs.

## Question

Can training improve a small local fable generator over a strong Instruct base
model on a constrained English children's-fable task?

## Fixed Task

All models generate from the same five narrative slots:

- character
- setting
- challenge
- outcome
- teaching / moral

The benchmark prompts are fixed in `data/test_prompts.jsonl`.

## Model Tracks

1. Base Instruct: `base-llama32-3b-instruct`
   - No training.
   - Expected strength: fluency and coherence.
   - Expected weakness: may ignore the required moral format.

2. SFT Llama 3.2 3B: `sft-llama32-3b-clean3k`
   - Fine-tuned on 3K cleaned synthetic fable examples.
   - Expected strength: stronger task format/adherence.
   - Expected weakness: possible fluency regression from imperfect data.

3. Base + Repair: not generated yet.
   - Use the same base model, then apply deterministic checks and a rewrite pass
     only for failed outputs.
   - Expected strength: keep base fluency while fixing empty `Moral:` and
     incomplete endings.
   - Expected weakness: this is an inference-time reliability method, not a new
     trained model.

4. Optional failure-focused LoRA: not trained yet.
   - Fine-tune only on examples targeting observed failures: empty moral,
     missing ending, weak slot adherence.
   - This is intentionally different from broad synthetic SFT, from-scratch SLM,
     DPO, or ORPO.

## Evaluation

Use the same prompt set for every model.

Automatic metrics:

- success rate
- `Moral:` footer rate
- exact character phrase rate
- exact moral keyword rate
- outcome coverage rate
- clean ending rate
- average length and latency

Human evaluation:

- English fluency
- prompt adherence
- fable structure
- moral clarity
- child safety

Final conclusion should be based on both automatic metrics and human scoring.
Do not claim a fine-tuned model is better if it only improves format while
damaging fluency.

## Current Clean Outputs

- Base: `results/baseline_outputs.jsonl`
- SFT Clean 3K: `results/sft_clean3k_outputs.jsonl`
- Human scores: `results/human_eval_base_vs_clean3k_10.md`
- Summary: `results/EXPERIMENT_RESULTS_SUMMARY.md`

Older experimental outputs were moved to `results/archive/pre_flow_reset_20260717`.

## Commands

Generate outputs:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --model-id base-llama32-3b-instruct --out-jsonl results\baseline_outputs.jsonl --out-md results\baseline_outputs.md
.\.venv\Scripts\python.exe scripts\run_baseline.py --model-id sft-llama32-3b-clean3k --out-jsonl results\sft_clean3k_outputs.jsonl --out-md results\sft_clean3k_outputs.md
```

Automatic evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_outputs.py --input results\baseline_outputs.jsonl --model-name "Base Llama 3.2 Instruct" --out-json results\eval_base.json --out-md results\eval_base.md
.\.venv\Scripts\python.exe scripts\evaluate_outputs.py --input results\sft_clean3k_outputs.jsonl --model-name "SFT Clean 3K" --out-json results\eval_sft_clean3k.json --out-md results\eval_sft_clean3k.md
```

Human evaluation template:

```powershell
.\.venv\Scripts\python.exe scripts\make_human_eval_template.py --limit 10 --model "Base=results\baseline_outputs.jsonl" --model "SFT Clean 3K=results\sft_clean3k_outputs.jsonl" --out results\human_eval_base_vs_clean3k_10.csv --out-md results\human_eval_base_vs_clean3k_10.md
```
