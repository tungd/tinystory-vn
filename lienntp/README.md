# lienntp Contribution Package

This folder summarizes the work contributed for the English fable generation project.

## Scope

The contribution focuses on practical evaluation and reliability improvement for local LLM fable generation:

- App integration for comparing multiple models.
- Flexible generation modes: `Raw`, `Postprocess`, and `Repair`.
- Dataset preparation and filtering pipelines.
- LoRA experiment artifacts and Ollama Modelfiles.
- Automatic metrics, human evaluation summaries, and visual figures.

## Non-App Contribution Snapshot

All non-app artifacts related to this contribution are also copied into:

```text
lienntp/full/
```

This snapshot includes:

- `scripts/`
- `docs/`
- `data/`
- `results/`
- `ollama/`

The root project files are still kept in place because the app imports and runtime paths depend on the normal project layout. App-related code remains at root:

- `app/`
- `web/`
- `config/`
- app tests under `tests/`

Use `lienntp/full/` for reviewing the non-app experiment assets, and use the root project to run the app.

## App Code Kept In Root

The runnable app code remains in the root project folders so imports, tests, and the web app do not break:

- `app/main.py`: generation endpoint with `raw`, `postprocess`, and `repair` modes.
- `app/enhanced_generation.py`: validation, moral post-processing, and repair logic.
- `app/config.py`: model and repair-model configuration.
- `config/models.json`: model registry for base and fine-tuned models.
- `web/src/components/InputPanel.tsx`: mode selector in the UI.
- `web/src/components/ObservabilityPanel.tsx`: displays generation mode and repair actions.
- `web/src/api.ts`: API metadata types for generation modes.

These web files are referenced from root `web/` and are not duplicated in `lienntp`.

See `lienntp/code/CODE_MANIFEST.md` for details.

## Data

Large training JSONL files are kept in the root `data/` directory to avoid duplicating data:

- `data/fluency_sft_v1/train.jsonl`
- `data/fluency_sft_v1/valid.jsonl`
- `data/failure_focused_lora_300/train.jsonl`
- `data/failure_focused_lora_300/valid.jsonl`
- `data/test_prompts.jsonl`

This folder keeps lightweight manifests/reports:

- `data/fluency_sft_v1_manifest.json`
- `data/fluency_sft_v1_dataset_report.md`
- `data/failure_focused_lora_300_manifest.json`
- `data/tf1_sft_500_manifest.json`

## Scripts

Important scripts are copied into `lienntp/scripts/` for review:

- `prepare_fluency_sft_dataset.py`
- `prepare_tf1.py`
- `build_failure_focused_lora.py`
- `run_baseline.py`
- `run_enhanced_base.py`
- `run_strict_prompt.py`
- `evaluate_outputs.py`
- `make_human_eval_template.py`
- `make_eval_figures.py`

The canonical runnable versions also remain in root `scripts/`.

## Models

The actual GGUF model files are not committed because they are large. The Ollama Modelfiles are included:

- `models/Modelfile.llama32-clean3k`
- `models/Modelfile.llama32-failure-lora`
- `models/Modelfile.llama32-fluency-sft-v1`

Local Ollama model names used during experiments:

- `llama32-fable-clean3k:q4`
- `llama32-fable-failure-lora:q4`
- `llama32-fable-fluency-sft-v1:q4`

## Results

Main reports:

- `results/FINAL_EXPERIMENT_REPORT.md`
- `results/EXPERIMENT_RESULTS_SUMMARY.md`

Visual figures:

- `results/figures/automatic_reliability.svg`
- `results/figures/human_eval_final_5way.svg`
- `results/figures/human_eval_fluency_sft_v1.svg`

## Key Finding

The tested LoRA fine-tunes did not outperform the base-oriented reliability pipeline. The best overall system is `Base + Repair`, while `Strict + Postprocess` is the strongest lightweight mode. This supports the project claim that prompt control plus deterministic/error-triggered correction can outperform limited fine-tuning for this local fable-generation task.
