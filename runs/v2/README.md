# v2 — 64M Metaspace model

- Model: GPT-2 style, 63.0M parameters (`768/7/12`)
- Data: 200,000 TF1-EN-3M fables
- Tokenizer: Metaspace BPE, vocabulary 16,384
- Training: 6,250 steps / 2 epochs
- Final loss: 1.73

## Local inventory

- `artifacts/hf/`: final HF model export; no local optimizer/scheduler state.
- `artifacts/mlx/`: current local inference model.
- `data/`: 200k training dataset and fixed tokenizer.
- `results/eval_summary.json`: canonical v2 evaluation.
- `results/live-gemma-eval-2026-07-16.json`: later end-to-end MLX → Gemma
  evaluation that exposed conditioning and stopping weaknesses.
- `logs/train-ephemeral.log`: first 9,375-step attempt, reclaimed around step 5,808.
- `logs/train-drive-resume.log`: Drive-backed 6,250-step run, reclaimed after
  checkpointing; final model/results were recovered from Drive.
- `logs/drive-eval-file.json`: Drive lookup record for the final evaluation.

The complete resumable checkpoints remain on Google Drive under
`MyDrive/fable200m_v2/ckpt/`, including `checkpoint-6250`.
