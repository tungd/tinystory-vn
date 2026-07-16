# v4 — wider, cleaner conditioned continuation (full evaluated)

v4 continues from v3-full weights with a fresh optimizer. It does not restart
the 63M model: architecture, tokenizer, and control format are unchanged. The
efficient improvement is one epoch over new data at a lower `5e-5` learning
rate, not more epochs over v3's old slice.

TF1-EN-3M uses one generator (Llama-3.1-8B-Instruct). Diversity therefore comes
from wider character/setting/challenge/outcome combinations, not generator
mixing.

## Data

v4 excludes:

- valid indices 0–199,999 used by v2;
- valid indices 200,000–200,099 used for v3 evaluation.

It scans from valid index 200,100 and accepts 250,000 new examples:

| Item | Count |
|---|---:|
| Source rows scanned | 525,508 |
| Accepted | 250,000 |
| Train | 245,076 |
| Validation | 4,924 |
| Unique characters | 8,561 |
| Settings / challenges / outcomes | 100 / 100 / 100 |

Quality filters:

- exact requested character appears within the first 120 words;
- at least one content word from the moral appears in the final 100 story words;
- 160–380 words after cleanup;
- duplicate prompt hashes removed;
- Markdown wrappers and existing moral footers removed;
- exactly one canonical `Moral: {dataset moral}` appended.

The 100 fresh evaluation controls begin at valid index 725,608 and are outside
all v2/v3/v4 training data.

## Training

| Phase | Examples | Steps | Base | Learning rate |
|---|---:|---:|---|---:|
| Pilot | 20,000 | 313 | v3-full | 5e-5 |
| Full | 245,076 | 3,830 | v3-full | 5e-5 |

Both phases start from immutable v3-full weights with fresh optimizer/scheduler
state. The pilot is diagnostic; the full run does not resume from it.

Runbook: `docs/runbooks/v4-train.md`.

## Pilot

The 20,000-example pilot completed 313 steps in 71.6 seconds: train loss 1.594,
final eval loss 1.593. On 20 fresh controls, exact character improved from 75%
to 80%; exact moral stayed 85% and both-exact stayed 70%. Strict Gemma judging
on five paired controls regressed from 3.40 to 3.10 overall, so the pilot was
mixed rather than a promotion result.

## Full run

The full run restarted from immutable v3-full weights and trained all 245,076
rows for one epoch:

- Runtime: 731.1 seconds
- Steps: 3,830 at 5.24 steps/second
- Train loss: 1.566
- Final fixed-subset eval loss: 1.546
- Drive model: `/MyDrive/fable200m_v4/full/hf`

On 100 fresh matched controls:

| Metric | v3 full | v4 |
|---|---:|---:|
| Exact character | 71% | 71% |
| Exact moral | 91% | 88% |
| Both exact | 65% | 64% |
| Clean `</story>` ending | 100% | 100% |

Strict Gemma judging on a seeded paired 20-control subset:

| Axis | v3 full | v4 |
|---|---:|---:|
| Grammar | 4.25 | 4.75 |
| Creativity | 2.80 | 3.00 |
| Moral clarity | 2.10 | 2.35 |
| Prompt adherence | 3.65 | 3.75 |
| Overall | 3.20 | 3.46 |

v4 modestly improves judged prose quality but does not improve deterministic
conditioning. It is not a clear replacement for v3. Unedited low/median/high
outputs are in `results/quality_samples.md`.
