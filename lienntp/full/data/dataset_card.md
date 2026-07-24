# Training Dataset Notes

This folder separates English training data from fixed evaluation prompts.

## Files

- `train.jsonl`: current promoted supervised fine-tuning split in English.
- `valid.jsonl`: current promoted validation split in English.
- `test_prompts.jsonl`: fixed evaluation prompts. Do not use these for training.
- `tf1/sft_100/`: 100-row TF1 subset for quick SFT smoke tests.
- `tf1/sft_500/`: 500-row TF1 subset promoted to `train.jsonl` / `valid.jsonl`.
- `tf1/sft_2000/`: 2000-row TF1 subset for a stronger SFT run.
- `fable_train_valid_tf1_sft500.zip`: zip with the promoted `train.jsonl` and `valid.jsonl` for Colab upload.

## Record Format

Each training and validation row uses:

```json
{
  "instruction": "Write a short English fable for children with a clear moral.",
  "input": "Main character: ...\nSetting: ...\nChallenge: ...\nOutcome: ...\nTeaching/Moral: ...",
  "output": "..."
}
```

## Evaluation Rule

Future fine-tuned models should be compared on the same English `test_prompts.jsonl` prompts against the baseline in `results/baseline_outputs.jsonl`.

## Rebuild Command

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe scripts\prepare_tf1.py --source hf --sizes 100,500,2000 --out data\tf1 --promote-size 500
Compress-Archive -Path data\train.jsonl,data\valid.jsonl -DestinationPath data\fable_train_valid_tf1_sft500.zip -Force
```

The preparation script streams `klusai/ds-tf1-en-3m`, parses the TF1 prompt into `Character`, `Setting`, `Challenge`, `Outcome`, and `Teaching`, filters out harsher topics for ages 4-7, and writes deterministic 90/10 train/valid splits.
