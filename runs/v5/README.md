# v5 — public-domain quality pilot

Status: source annotation in progress.

v5 tests whether a small amount of coherent human prose improves the 63M model
more efficiently than another large TF1 continuation. It starts from immutable
v3-full weights with a fresh optimizer; v4 is not promoted.

## Data policy

- Nine human-authored Project Gutenberg fable/folk-tale collections.
- Raw books and prepared data remain gitignored; tracked metadata keeps exact
  Gutenberg IDs and URLs.
- Gemma labels an exact protagonist phrase, demonstrated trait, and causal
  moral. It may reject a section but never rewrites the story.
- Accepted human stories remain unrewritten apart from whitespace normalization
  and the canonical `Moral: ...` footer required by the existing target format.
- `demelin/understanding_fables` remains evaluation-only and is not training
  data.

The parser found 394 complete sections between 70 and 450 words. A stratified
45-story annotation smoke test accepted 36 (80%) after retrying free-tier API
rate limits.

## Pilot mixture

- Hash-hold out 10% of accepted human stories for validation.
- Repeat each remaining human story three times.
- Add one deterministic v3 replay example per unique human training story.
- Train ten short epochs at `1e-5`, expected about 180 total steps.

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

Runbook: `docs/runbooks/v5-train.md`.
