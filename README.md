# English Fable Generator

A local web app for generating short English fables for children with Ollama, FastAPI, and React.

Experiment assets, datasets, training/evaluation scripts, reports, and figures from the `lienntp` contribution are kept in:

```text
lienntp/
```

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

## Experiments

Training/data-processing scripts, datasets, model Modelfiles, evaluation tables, reports, and visual figures are intentionally not mixed into the app root. See:

```text
lienntp/README.md
lienntp/results/FINAL_EXPERIMENT_REPORT.md
lienntp/results/figures/
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
cd web
npm.cmd run build
```
