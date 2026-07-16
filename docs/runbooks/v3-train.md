# v3 continuation runbook

Goal: strengthen exact character/moral conditioning without retraining the 63M
model from scratch. v2 remains immutable. v3 loads only v2 weights; optimizer
and scheduler start fresh.

## Local preparation check

```bash
uv sync --extra dev --extra inference --extra colab
uv run python scripts/prepare_v3.py
uv run python scripts/train_v3.py --dry-run
```

Expected: 199,688 accepted rows, then `status: ready`. Generated data is under
`runs/v3/data/` and gitignored. Each checkpoint evaluates a fixed 1,024-row
subset; the full 9,984-row split remains available for final evaluation.

## Colab setup

v2 must already exist at `/content/drive/MyDrive/fable200m_v2/` with
`fables.jsonl`, `tokenizer.json`, and the final HF model under `ckpt/`.

```bash
colab new -s v3 --gpu A100
colab drivemount -s v3

echo "import os; os.makedirs('/content/scripts', exist_ok=True)" | colab exec -s v3
colab upload -s v3 scripts/prepare_v3.py /content/scripts/prepare_v3.py
colab upload -s v3 scripts/train_v3.py /content/scripts/train_v3.py
colab upload -s v3 scripts/prepare_tf1.py /content/scripts/prepare_tf1.py
colab upload -s v3 scripts/fable_tokenizer.py /content/scripts/fable_tokenizer.py
```

Install dependencies and prepare v3 directly on mounted Drive:

```bash
echo "import subprocess; subprocess.run(['pip','install','-q','transformers>=4.44','datasets>=3.0','accelerate>=1.0'], check=True)" | colab exec -s v3 --timeout 300

echo "import subprocess; subprocess.run(['python3','/content/scripts/prepare_v3.py','--source','/content/drive/MyDrive/fable200m_v2/fables.jsonl','--tokenizer','/content/drive/MyDrive/fable200m_v2/tokenizer.json','--out','/content/drive/MyDrive/fable200m_v3/data'], check=True)" | colab exec -s v3 --timeout 300

echo "import subprocess; subprocess.run(['python3','/content/scripts/train_v3.py','--data','/content/drive/MyDrive/fable200m_v3/data','--base-model','/content/drive/MyDrive/fable200m_v2/ckpt','--out','/content/drive/MyDrive/fable200m_v3/pilot/hf','--dry-run'], check=True)" | colab exec -s v3 --timeout 300
```

## Pilot first

Launch detached so a CLI timeout cannot stop training:

```bash
echo "import os,subprocess; os.makedirs('/content/drive/MyDrive/fable200m_v3/logs',exist_ok=True); log=open('/content/drive/MyDrive/fable200m_v3/logs/pilot.log','w'); cmd=['python3','/content/scripts/train_v3.py','--data','/content/drive/MyDrive/fable200m_v3/data','--base-model','/content/drive/MyDrive/fable200m_v2/ckpt','--out','/content/drive/MyDrive/fable200m_v3/pilot/hf','--train-samples','20000','--max-steps','500','--warmup-steps','50']; p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print('pid',p.pid)" | colab exec -s v3
```

Inspect `/content/drive/MyDrive/fable200m_v3/logs/pilot.log`. Continue only if
validation loss improves and sampled stories state the exact moral while using the
requested character. Judge v2 and pilot on identical controls.

## Full one-epoch continuation

Use a new output directory; never resume the pilot optimizer:

```bash
echo "import os,subprocess; log=open('/content/drive/MyDrive/fable200m_v3/logs/full.log','w'); cmd=['python3','/content/scripts/train_v3.py','--data','/content/drive/MyDrive/fable200m_v3/data','--base-model','/content/drive/MyDrive/fable200m_v2/ckpt','--out','/content/drive/MyDrive/fable200m_v3/full/hf','--train-samples','0','--max-steps','2965','--warmup-steps','100']; p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print('pid',p.pid)" | colab exec -s v3
```

Stop the VM after artifacts/logs are confirmed on Drive:

```bash
colab stop -s v3
```

## Acceptance gate

- Use fresh TF1 controls beyond the original first 200,000 valid rows; the v3
  validation split is not unseen by v2.
- Same fixed controls and generation settings for v2/v3.
- Requested character appears in every generated story.
- Exact requested moral is explicit near the ending.
- Gemma judge improves moral clarity and prompt adherence without material
  grammar/creativity regression.
- Store generations and judge JSON under `runs/v3/results/` after download.
