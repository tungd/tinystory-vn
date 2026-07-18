# English Fable Generator

A local web app for generating short English fables for children with Ollama, FastAPI, and React. The project is designed for practical model comparison: start with a base Instruct model, then add fine-tuned or alternative base models and evaluate them on the same prompt set.

## Current Default Model

Use an Instruct/chat model, not a thinking model.

Default registry entry:

```json
{
  "id": "base-llama32-3b-instruct",
  "name": "Llama 3.2 3B Instruct Q4",
  "ollama": "llama3.2:3b",
  "kind": "base"
}
```

Thinking models can leak reasoning text into the generated story. Keep `FABLE_THINK=false`.
The default model is the installed Q4 Ollama model for practical local testing.

## Requirements

- Python 3.11+
- Node.js 20+
- Ollama
- At least one local Ollama chat/Instruct model

On this machine, Ollama models are stored in:

```text
D:\OllamaModels
```

## Install

```powershell
cd D:\NgLin\NgLin\ThS\Ky4\GenAI\PJ\tinystory-vn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install/build the frontend:

```powershell
cd web
npm.cmd install
npm.cmd run build
cd ..
```

Pull the default model if needed:

```powershell
ollama pull llama3.2:3b
```

## Run

Make sure Ollama is running, then start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

If port `8000` is busy, use another port:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

## App Workflow

- Playground: generate a single English fable from five optional narrative fields.
- Compare: run two configured models side by side.
- Results: show batch evaluation from `results/eval_summary.json` when that file exists.

Narrative fields:

- Main character
- Setting
- Challenge
- Outcome
- Teaching/Moral

Generation modes:

- `Raw`: direct model generation.
- `Post`: direct generation plus deterministic final `Moral:` line normalization.
- `Repair`: validates the draft, rewrites severe failures with the configured repair model, then normalizes the final moral line.

The default repair model id is controlled by `FABLE_REPAIR_MODEL_ID` and defaults to:

```text
sft-llama32-3b-clean3k
```

## Model Configuration

Edit `config/models.json`. The `ollama` value must match a model shown by:

```powershell
ollama list
```

Add fine-tuned models as additional entries:

```json
{
  "id": "sft-llama32-3b",
  "name": "Llama 3.2 3B SFT",
  "ollama": "llama32-fable-sft",
  "kind": "finetuned",
  "desc": "Fine-tuned on English fable data"
}
```

Current experiment models include:

- `base-llama32-3b-instruct`: installed local Llama 3.2 3B Instruct Q4 baseline.
- `base-llama32-3b-instruct-q4`: local Q4 base used for strict-prompt ablations.
- `sft-llama32-3b-clean3k`: broad SFT contrast model.
- `failure-lora-llama32-3b`: 300-row failure-focused LoRA.
- `fluency-sft-v1-lora-q4`: 10K quality-filtered fluency SFT LoRA.

## Batch Generation

Generate outputs on fixed test prompts:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --model-id base-llama32-3b-instruct --out-jsonl results\baseline_outputs.jsonl --out-md results\baseline_outputs.md
```

Create a human evaluation CSV:

```powershell
.\.venv\Scripts\python.exe scripts\make_human_eval_template.py --limit 10
```

Compare two batch output files:

```powershell
.\.venv\Scripts\python.exe scripts\compare_model_outputs.py
```

## Dataset Preparation

The project uses fixed test prompts plus TF1-derived SFT splits for training experiments:

- `data/tf1/sft_100`: quick pipeline smoke test.
- `data/tf1/sft_500`: default promoted train/valid split.
- `data/tf1/sft_2000`: larger SFT run.

Rebuild the TF1 subsets:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe scripts\prepare_tf1.py --source hf --sizes 100,500,2000 --out data\tf1 --promote-size 500
Compress-Archive -Path data\train.jsonl,data\valid.jsonl -DestinationPath data\fable_train_valid_tf1_sft500.zip -Force
```

Use the same `data/test_prompts.jsonl` for every model so baseline, SFT-100, SFT-500, SFT-2000, and later ORPO runs are directly comparable.

Additional datasets prepared during the final experiments:

- `data/failure_focused_lora_300.zip`: 300 targeted examples for moral/footer and structure failures.
- `data/fluency_sft_v1/train.jsonl`: 9,000 training rows from quality-filtered TF1 fables.
- `data/fluency_sft_v1/valid.jsonl`: 1,000 validation rows.
- `data/fluency_sft_v1/dataset_report.md`: filtering report for the 10K fluency dataset.

## Evaluation

Use the same `data/test_prompts.jsonl` for all models. Recommended criteria:

- English fluency
- Prompt adherence
- Fable structure
- Moral clarity
- Child safety

For the final report, compare base vs fine-tuned models using both automatic metrics and human or LLM-as-judge scores.

## Final Experiment Summary

The final report is saved at:

```text
results/FINAL_EXPERIMENT_REPORT.md
```

Main systems tested:

- Base FP16
- SFT Clean 3K
- Failure-Focused LoRA 300
- Strict Prompt
- Strict + Postprocess
- Base + Repair
- Fluency SFT v1 LoRA 10K

Key human-evaluation result:

| Model | Fluency | Adherence | Structure | Moral | Safety | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base + Repair | 3.40 | 4.30 | 4.10 | 5.00 | 4.80 | 4.32 |
| Strict + Postprocess | 3.70 | 4.10 | 3.60 | 5.00 | 4.60 | 4.20 |
| Fluency SFT v1 | 3.00 | 3.70 | 4.00 | 3.70 | 4.80 | 3.84 |

Conclusion:

- Base + Repair is the best overall system.
- Strict + Postprocess is the strongest lightweight mode and has the best fluency in the final three-way comparison.
- Fluency SFT v1 did not improve fluency overall despite using a 10K filtered dataset.
- The main contribution is evaluation-driven reliability improvement for local fable generation: prompt control plus deterministic/error-triggered correction outperformed the tested LoRA fine-tunes under the available data and compute limits.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
cd web
npm.cmd run build
```
