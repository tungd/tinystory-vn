# Context

This project targets short English fables for children ages 4-7.

## Target Output

Each story should be:

- English
- Short and coherent
- Age-appropriate
- Fable-like, usually with an animal character
- Clear about its moral lesson

## Input Structure

The app uses five optional narrative fields:

- Main character
- Setting
- Challenge
- Outcome
- Teaching/Moral

Empty fields are skipped and the model fills in the missing details.

## Model Direction

Use chat/Instruct models. Avoid thinking models because they may include reasoning text in the final story.

Current default:

```text
base-llama32-3b-instruct -> llama3.2:3b
```

The app supports three generation modes:

- raw
- postprocess
- repair

The final report is in `results/FINAL_EXPERIMENT_REPORT.md`.

## Evaluation Axes

- Grammar / English fluency
- Creativity
- Moral clarity
- Prompt adherence
- Child safety

The main comparison should be base model vs fine-tuned or alternative model on the same fixed prompt set.
