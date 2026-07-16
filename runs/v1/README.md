# v1 — 29M baseline

- Model: GPT-2 style, 29.9M parameters (`512/8/8`)
- Data: 200,000 TF1-EN-3M fables
- Tokenizer: raw BPE, vocabulary 8,192 (known broken-word bug)
- Training: 9,375 steps / 3 epochs
- Final loss: 1.87

## Local inventory

- `artifacts/hf/`: complete resumable HF checkpoint, including optimizer,
  scheduler, RNG, and trainer state.
- `artifacts/mlx/`: converted inference model.
- `data/`: 200k training dataset and tokenizer.
- `pilot-15k/`: earlier 15k pipeline-validation run.
- `results/eval_summary.json`: canonical v1 evaluation restored from commit
  `97673fe`.
- `incoming/`: preserved duplicate downloads. Their hashes exactly match
  `artifacts/hf/model.safetensors` and `artifacts/hf/optimizer.pt`.

No complete raw v1 training log was found locally. Its verified training and
recovery timeline was reconstructed from the SCV transcript in
`logs/transcript-recovery.md`.
