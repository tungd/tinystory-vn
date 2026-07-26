# No-retraining ablation summary

## Condition availability

| Model | Mode | Slot coverage /5 | Added slots /3 | Requested causal |
|---|---|---:|---:|---:|
| e2 | full | 0.48 | 0.32 | 1.44 |
| e2 | two_slot | 0.04 | 0.04 | 1.00 |
| e5 | full | 4.88 | 2.96 | 9.24 |
| e5 | two_slot | 1.20 | 0.16 | 1.96 |

## Counterfactual sensitivity

| Model | Match both | Intervention changes story | Sensitivity /10 |
|---|---:|---:|---:|
| e2 | 0.10 | 0.10 | 1.70 |
| e5 | 1.00 | 1.00 | 9.40 |

## E3 repair

| Mode | Exact moral | Moral footer | Requested causal |
|---|---:|---:|---:|
| raw | 0.20 | 0.20 | 9.00 |
| repaired | 1.00 | 1.00 | 9.64 |
