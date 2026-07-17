# v5 human-story quality pilot

Goal: test a small quality-weighted continuation from v3-full. Never initialize
from v4; it did not pass the conditioning gate.

## Source and annotate

```bash
uv run python scripts/v5_sources.py --max-words 450

uv run --env-file .env python scripts/label_v5.py --workers 2 --request-interval 4
```

Source preparation also fetches the MIT-licensed `Understanding Fables` manual
paraphrases and reserves a deterministic 20% external holdout. Annotation is
resumable. API errors are retried on the next invocation; latest successful
annotation wins for each source.

Build the quality-weighted mixture:

```bash
uv run python scripts/prepare_v5.py
uv run --extra colab python scripts/train_v5.py \
  --base-model runs/v2/artifacts/hf --dry-run
```

The local dry-run uses v2 only for architecture/tokenizer compatibility. Actual
training loads immutable v3-full from Drive.

## Colab

Use the persistent Drive authorization workflow in `AGENTS.md`. Package
`runs/v5/data/prepared` as a split archive, mount Drive, upload these scripts,
and install the same dependencies as v4:

- `train_v3.py`
- `train_v5.py`
- `generate_v3_comparison.py`
- `generate_v5_comparison.py`

Train:

```bash
python3 /content/scripts/train_v5.py \
  --data /content/v5-data/prepared \
  --base-model /content/drive/MyDrive/fable200m_v3/full/hf \
  --out /content/drive/MyDrive/fable200m_v5/pilot/hf
```

Frozen data: 1,632 mixed examples and 260 steps.

Do not assume the final checkpoint is best. Generate 20 matched controls from
`checkpoint-50`, `checkpoint-100`, `checkpoint-150`, `checkpoint-200`,
`checkpoint-250`, and the final export. Pick the duration using deterministic
conditioning plus manual coherence inspection, then run the selected checkpoint
through the full evaluation below.

## Evaluation

Generate 100 matched outputs:

```bash
python3 /content/scripts/generate_v5_comparison.py \
  --v3 /content/drive/MyDrive/fable200m_v3/full/hf \
  --v5 /content/drive/MyDrive/fable200m_v5/pilot/hf/checkpoint-50 \
  --controls-file /content/v5-data/prepared/eval_controls.json \
  --out /content/drive/MyDrive/fable200m_v5/results/pilot_generations_100.json
```

Local deterministic metrics:

```bash
uv run python scripts/evaluate_v5_comparison.py \
  --input runs/v5/results/pilot_generations_100.json \
  --out runs/v5/results/pilot_metrics_100.json
```

Blind strict-schema paired judge. `minimal` is intentional: high thinking
exhausted both 2,000 and 8,192-token output budgets without completing JSON.

```bash
FABLE_JUDGE_THINKING_LEVEL=minimal uv run --env-file .env \
  python scripts/judge_v5_pairwise.py \
  --input runs/v5/results/pilot_generations_100.json \
  --out runs/v5/results/pilot_pairwise_judged_20.json \
  --max-output-tokens 2000
```

Stop the Colab VM only after model, generations, and logs exist on Drive and
results have been downloaded locally.
