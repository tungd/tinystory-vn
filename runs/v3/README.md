# v3 — conditioning-focused continuation (pilot complete)

v3 is isolated from v1/v2. It initializes from v2 model weights, never its
optimizer/scheduler state.

Implemented changes:

- Preserve each exact TF1 character/moral pair in the control prefix.
- Append `Moral: {moral}` before `</story>` in every training target.
- Mask the `<char>/<moral>/<story>` prefix from loss.
- Retain the v2 Metaspace tokenizer and 63M architecture.
- Require the complete requested character phrase in the original story.
- Run a 20k-example pilot before the one-epoch continuation.

Local preparation and verification:

```bash
uv run --extra colab python scripts/prepare_v3.py
uv run --extra colab python scripts/train_v3.py --dry-run
uv run pytest -q
```

Prepared data: 108,487 accepted, 91,513 rejected; 103,063 train and 5,424
validation examples, seed 42. Training loss checks use a fixed 1,024-row
validation subset. Bulk data stays gitignored.

That split was seen during v2 training. Final v2-v3 judgment must use fresh TF1
rows after the original first 200,000 valid source rows.

Canonical destinations:

- `data/`: reformatted v3 dataset and fixed held-out split.
- `artifacts/hf/`, `artifacts/mlx/`: v3-only outputs.
- `logs/`: preparation, pilot, and full-run logs.
- `results/`: v3 evaluation plus v2-v3 comparison.

Actual Colab commands: `docs/runbooks/v3-train.md`.

Pilot result (20,000 examples, 313 steps): train loss 1.509, final eval loss
1.502, about 71 seconds on A100. The pilot fixed clean stopping (4/4 versus 0/4
for v2), but exact character and moral matches remained 2/4 each.

Native Gemma judge result (`gemma-4-26b-a4b-it`, minimal thinking):

- v2 overall: 4.75
- v3 pilot overall: 5.25
- Mean judge latency: 5.64 seconds over eight calls
- Full generations and rationales: `results/`

The four-control pilot is directional, not the final held-out evaluation.
