# Versioned Runs

Each training generation owns its data, artifacts, logs, results, and manifest:

```text
runs/
├── v1/  # 29.9M, raw BPE, 3 epochs
├── v2/  # 63.0M, Metaspace BPE, 2 epochs
└── v3/  # planned conditioning-focused continuation from v2
```

Large weights, datasets, optimizer state, and raw logs remain local/ignored.
Run manifests, evaluation results, documentation, and MLX metadata are tracked.

Compatibility symlinks under `models/`, `data/`, and `results/` preserve the
old paths used by scripts and the app. New work should use `runs/<version>/`.
