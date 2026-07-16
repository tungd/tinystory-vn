# v5 — public-domain quality pilot

Status: source annotation in progress.

v5 tests whether a small amount of coherent human prose improves the 63M model
more efficiently than another large TF1 continuation. It starts from immutable
v3-full weights with a fresh optimizer; v4 is not promoted.

## Data policy

- Nine human-authored Project Gutenberg fable/folk-tale collections.
- 189 MIT-licensed `demelin/understanding_fables` stories manually paraphrased
  into contemporary English; deterministic 80% train / 20% external holdout.
- Raw books and prepared data remain gitignored; tracked metadata keeps exact
  Gutenberg IDs and URLs.
- Gemma labels an exact protagonist phrase, demonstrated trait, and causal
  moral. It may reject a section but never rewrites the story.
- Preparation deterministically inserts that one trait at the first exact
  protagonist mention, then adds the canonical `Moral: ...` footer. No other
  prose is generated or rewritten.
- Prepared Gutenberg stories are 70-250 words; concise modern paraphrases may be
  50-250 words. Their causal endings fit the existing generation budget.
- The modern source supplies its exact moral; Gemma cannot replace it.

The parser found 398 Gutenberg sections between 70 and 450 words, plus 189
modern paraphrases. A stratified 45-story Gutenberg annotation smoke test
accepted 36 (80%) after retrying free-tier API rate limits.

## Pilot mixture

- Hash-hold out 10% of accepted human stories for validation.
- Repeat each remaining human story three times.
- Add one deterministic v3 replay example per unique human training story.
- Train ten short epochs at `1e-5`, expected about 180 total steps.

Checkpoints at steps 50, 100, 150, and final are a built-in duration sweep.
Screen all four on 20 matched controls before running the 100-control evaluation;
the final checkpoint is not automatically preferred.

The repeated human set supplies 75% of training rows. Replay limits catastrophic
forgetting; it is not counted as quality data.

## Evaluation gate

Use the same 100 v4-held-out TF1 controls for direct v3 comparison:

- Exact character at least 70%.
- Exact moral at least 90%.
- Clean ending 100%.
- Blind, randomized, high-thinking paired Gemma judge prefers v5 over v3-full.
- Mean pairwise quality improves at least 0.5 without grammar or moral-clarity
  regression.
- Record unedited low/median/high stories before promotion.

Then run the selected checkpoint on the reserved modern controls as a secondary
out-of-source check. These do not replace the direct matched TF1 comparison.

Runbook: `docs/runbooks/v5-train.md`.
