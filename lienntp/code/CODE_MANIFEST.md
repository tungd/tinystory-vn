# Code Manifest

App-related code remains in the root project so the app, tests, and imports continue to work. Non-app experiment code is kept under `lienntp/scripts/` and `lienntp/full/scripts/`.

## Backend

| File | Purpose |
| --- | --- |
| `app/main.py` | Adds app-level generation mode support: `raw`, `postprocess`, `repair`. |
| `app/enhanced_generation.py` | Validates generated stories, normalizes final moral lines, and optionally rewrites severe failures. |
| `app/config.py` | Adds repair-model configuration through `FABLE_REPAIR_MODEL_ID`. |
| `config/models.json` | Registers base, SFT, failure LoRA, and fluency SFT models. |

These backend app files are intentionally not copied here; they stay in root.

## Frontend

| File | Purpose |
| --- | --- |
| `web/src/components/InputPanel.tsx` | Adds UI mode selector for Raw/Post/Repair. |
| `web/src/components/ObservabilityPanel.tsx` | Shows mode and enhancement actions in runtime metadata. |
| `web/src/api.ts` | Adds generation-mode metadata types. |

Frontend files are intentionally kept in the root `web/` directory only, because the app build and backend static serving depend on that location.

## Scripts

| File | Purpose |
| --- | --- |
| `scripts/prepare_fluency_sft_dataset.py` | Builds the 10K quality-filtered fluency SFT dataset. |
| `scripts/build_failure_focused_lora.py` | Builds the 300-row failure-focused LoRA dataset. |
| `scripts/run_enhanced_base.py` | Runs postprocess/repair pipelines offline on fixed prompts. |
| `scripts/evaluate_outputs.py` | Computes automatic reliability metrics. |
| `scripts/make_human_eval_template.py` | Builds human-evaluation CSV/Markdown templates. |
| `scripts/make_eval_figures.py` | Generates SVG charts from automatic and human evaluation results. |

## Tests

| File | Purpose |
| --- | --- |
| `tests/test_enhanced_generation.py` | Tests postprocess/repair validation helpers. |
| `tests/test_api_en.py` | Covers API behavior after English-mode and model changes. |
| `tests/test_prepare_tf1.py` | Tests TF1 dataset preparation logic. |

## Verification

Latest verified commands:

```powershell
.\.venv\Scripts\python.exe -m pytest
cd web
npm.cmd run build
```

Expected:

```text
72 passed
frontend build: success
```
