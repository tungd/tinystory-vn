# v3 — conditioning-focused continuation (planned)

v3 is isolated from v1/v2. It will initialize from the v2 model weights while
resetting optimizer and scheduler state.

Planned changes:

- Use exact held-out TF1 character/moral pairs for baseline and evaluation.
- Append `Moral: {moral}` before `</story>` in every training target.
- Mask the `<char>/<moral>/<story>` prefix from loss.
- Retain the v2 Metaspace tokenizer and 63M architecture.
- Run a 20k-example pilot before a full 200k, one-epoch continuation.

Canonical destinations:

- `data/`: reformatted v3 dataset and fixed held-out split.
- `artifacts/hf/`, `artifacts/mlx/`: v3-only outputs.
- `logs/`: preparation, pilot, and full-run logs.
- `results/`: v3 evaluation plus v2-v3 comparison.
