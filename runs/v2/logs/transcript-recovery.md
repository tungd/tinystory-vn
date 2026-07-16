# v2 transcript recovery

Source authority: SCV session `prAfVcPU6lP5q6hhGYIZ0g`, local transcript
`~/.scv/transcripts/prAfVcPU6lP5q6hhGYIZ0g.jsonl`.

The raw transcript is not copied here because it is very large and contains
ephemeral authorization URLs. Relevant durable events:

| Seq | Recovered event |
|---:|---|
| 3103 | Drive `checkpoint-8500` proved corrupted/wrong (tiny weight file and old 29M config). Intact v2 fallback was local `checkpoint-2500`. |
| 3110 | Target reduced from 9,375 to 6,250 steps (2 epochs) to finish within one A100 session. |
| 3121–3126 | Checkpoint cadence tightened to every 250 steps. |
| 3201–3208 | Data/checkpoints moved to Drive; separate checkpoint directories chosen as stepped backups. |
| 3279 | Drive folder `fable200m_v2`, ID `1MDx9w7iVYdXove6xs8mUWaCbzbuxPGXj`, verified. |
| 3331–3395 | Drive data/root tokenizer repaired; resume source established at `checkpoint-2500`. |
| 3418 | Step 3631/6250, epoch 1.18, loss 1.73, ~1.7 it/s. |
| 3432–3441 | VM reclaimed; Drive persisted outputs; final `eval_summary.json` recovered. |
| 3450 | Completed v2 eval: Distinct-1 0.519, Distinct-2 0.922, Self-BLEU 0.028, Flesch 81.5. |
| 3472–3496 | 251,952,784-byte HF weights downloaded and converted to MLX with key remap plus Conv1D transpose. |
| 3528 | Longer generation exposed weak moral adherence, repetition, and weak story arc despite clean word boundaries. |
| 3531 | Proposed v3 data target: explicit moral ending plus additional training. |

The two raw training logs recovered from `/tmp` are preserved beside this note:
`train-ephemeral.log` and `train-drive-resume.log`.
