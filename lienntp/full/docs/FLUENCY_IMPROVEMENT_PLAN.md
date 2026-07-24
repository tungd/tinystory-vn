# Fluency Improvement Plan

Current best human-scored system: `Base + Repair` at 4.32/5 overall.

Current best fluency score among generated systems: `Strict Prompt` at 3.70/5,
but it underperforms on moral reliability.

## Next Candidate

Use `Strict Prompt + Postprocess`.

Rationale:

- Strict Prompt had the highest human fluency score in the 5-way evaluation.
- Strict Prompt also achieved strong automatic character control.
- Postprocess deterministically fixes missing/weak final `Moral:` formatting.
- This avoids another risky fine-tune and stays distinct from the two reference
  projects.

## Current Automatic Metrics

| Metric | Strict Prompt | Strict Prompt + Postprocess |
| --- | ---: | ---: |
| Success | 25/25 | 25/25 |
| Moral footer rate | 0.56 | 1.00 |
| Moral exact rate | 0.56 | 1.00 |
| Character exact rate | 0.92 | 0.92 |
| Outcome coverage rate | 0.80 | 0.80 |
| Clean ending rate | 0.96 | 1.00 |
| Run-on sentence rate | 0.00 | 0.00 |
| Avg latency ms | 21156 | 21156 |

## Experiment To Run

Human-score 10 prompts for:

1. `Base + Repair`
2. `Strict Prompt`
3. `Strict Prompt + Postprocess`

If `Strict Prompt + Postprocess` improves fluency while matching repair
reliability, it becomes the final recommended system. If not, keep
`Base + Repair` as final and report Strict+Postprocess as a promising ablation.

## If More Training Is Required

Do not run broad SFT again. A safer training direction is a small
fluency-preserving LoRA:

- Use only high-fluency targets.
- Keep final `Moral:` exactly formatted.
- Mix in base-like fluent stories as replay examples.
- Use lower learning rate than the failed Failure LoRA run.

This should only be attempted after the Strict+Postprocess human evaluation.
