# v3 — conditioning-focused continuation (prepared)

v3 is isolated from v1/v2. It initializes from v2 model weights, never its
optimizer/scheduler state.

Implemented changes:

- Preserve each exact TF1 character/moral pair in the control prefix.
- Append `Moral: {moral}` before `</story>` in every training target.
- Mask the `<char>/<moral>/<story>` prefix from loss.
- Retain the v2 Metaspace tokenizer and 63M architecture.
- Reject source stories that do not mention the requested character.
- Run a 20k-example pilot before the one-epoch continuation.

Local preparation and verification:

```bash
uv run --extra colab python scripts/prepare_v3.py
uv run --extra colab python scripts/train_v3.py --dry-run
uv run pytest -q
```

Prepared data: 199,688 accepted, 312 rejected; 189,704 train and 9,984
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
