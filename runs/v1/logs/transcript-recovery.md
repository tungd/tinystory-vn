# v1 transcript recovery

Source authority: SCV session `prAfVcPU6lP5q6hhGYIZ0g`, local transcript
`~/.scv/transcripts/prAfVcPU6lP5q6hhGYIZ0g.jsonl`.

The raw transcript is not copied here because it is very large and contains
ephemeral authorization URLs. Relevant durable events:

| Seq | Recovered event |
|---:|---|
| 1411–1416 | `checkpoint-500` verified with all ten resume files; 343 MB total; copied to Drive. |
| 1434 | Step 928/9375, loss 4.95, initial loss about 8.08, ~2.9 it/s. |
| 1456–1478 | Colab reclaimed; `checkpoint-2000` recovered from Drive and verified locally. |
| 1519 | Fresh A100 successfully resumed at step 2001 with optimizer state intact. |
| 1532 | Step 4325/9375, loss 2.15; checkpoint-3000 safely on Drive. |
| 1545 | Step 6258/9375, loss 1.88; checkpoint-5000 synced. |
| 1563–1581 | Final approach: steps 8238, 8663, then 9105/9375. |
| 1590 | Training completed: 29.9M params, 9,375 steps, 3 epochs, final loss 1.87. |
| 1590 | Eval: Distinct-1 0.389, Distinct-2 0.857, Self-BLEU 0.078, Flesch 82.9. |
| 1595 | `eval_summary.json` and final checkpoint downloaded; A100 stopped. |
| 1616 | Drive still held full model, optimizer, tokenizer, and config after local `/tmp` cleanup. |
| 1637 | v1 evaluation committed and pushed; later history identifies commit `97673fe`. |

Drive provenance recovered from the transcript: `MyDrive/fable200m/run1`
contained at least `checkpoint-2000` and `checkpoint-7500`. The complete local HF
checkpoint under `runs/v1/artifacts/hf/` supersedes these for local resume.
