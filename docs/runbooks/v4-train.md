# v4 continuation runbook

Goal: improve story quality and conditioning using 250,000 new, screened TF1
examples. v3-full stays immutable. Pilot and full runs each load only its model
weights and create fresh optimizer/scheduler state.

## Local data

Already prepared under gitignored `runs/v4/data/`:

```bash
uv run --extra colab python scripts/prepare_v4.py
uv run --extra colab python scripts/train_v4.py \
  --base-model runs/v2/artifacts/hf --dry-run
```

The local dry-run uses v2 only as an architecture/tokenizer-compatible config
check because v3-full remains on Drive. Actual training must use v3-full.

## Package data for Colab

Compressed data exceeds the Colab upload limit, so split it into 35 MiB parts:

```bash
tar -czf /tmp/v4-data.tgz -C runs/v4 data
split -b 35m -d -a 2 /tmp/v4-data.tgz /tmp/v4-data.tgz.part-
```

## Colab setup

```bash
colab new -s v4 --gpu A100
colab drivemount -s v4

echo "import os; os.makedirs('/content/scripts', exist_ok=True)" | colab exec -s v4
colab upload -s v4 scripts/train_v3.py /content/scripts/train_v3.py
colab upload -s v4 scripts/train_v4.py /content/scripts/train_v4.py
colab upload -s v4 scripts/generate_v3_comparison.py /content/scripts/generate_v3_comparison.py
colab upload -s v4 scripts/generate_v4_comparison.py /content/scripts/generate_v4_comparison.py
colab upload -s v4 scripts/prepare_tf1.py /content/scripts/prepare_tf1.py
colab upload -s v4 scripts/fable_tokenizer.py /content/scripts/fable_tokenizer.py
```

Upload every archive part:

```bash
for part in /tmp/v4-data.tgz.part-*; do
  colab upload -s v4 "$part" "/content/$(basename "$part")"
done
```

Reassemble and extract to Drive, then install dependencies:

```bash
echo "import glob,os; parts=sorted(glob.glob('/content/v4-data.tgz.part-*')); out=open('/content/v4-data.tgz','wb'); [out.write(open(p,'rb').read()) for p in parts]; out.close(); os.makedirs('/content/drive/MyDrive/fable200m_v4',exist_ok=True)" | colab exec -s v4

echo "import subprocess; subprocess.run(['tar','-xzf','/content/v4-data.tgz','-C','/content/drive/MyDrive/fable200m_v4'],check=True); subprocess.run(['pip','install','-q','transformers>=4.44','datasets>=3.0','accelerate>=1.0'],check=True)" | colab exec -s v4 --timeout 300
```

## Pilot

```bash
echo "import os,subprocess; root='/content/drive/MyDrive/fable200m_v4'; os.makedirs(root+'/logs',exist_ok=True); log=open(root+'/logs/pilot.log','w'); cmd=['python3','/content/scripts/train_v4.py','--data',root+'/data','--base-model','/content/drive/MyDrive/fable200m_v3/full/hf','--out',root+'/pilot/hf','--train-samples','20000','--save-steps','150']; p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print('pid',p.pid)" | colab exec -s v4
```

Pilot generation uses the first 20 fresh controls:

```bash
echo "import subprocess; root='/content/drive/MyDrive'; subprocess.run(['python3','/content/scripts/generate_v4_comparison.py','--v3',root+'/fable200m_v3/full/hf','--v4',root+'/fable200m_v4/pilot/hf','--controls-file',root+'/fable200m_v4/data/eval_controls.json','--controls','20','--out',root+'/fable200m_v4/results/pilot_generations_20.json'],check=True)" | colab exec -s v4 --timeout 600
```

Continue only if outputs remain coherent, exact conditioning is not worse than
v3-full, and every story stops cleanly.

## Full one-epoch run

Start again from v3-full, never from the pilot:

```bash
echo "import os,subprocess; root='/content/drive/MyDrive/fable200m_v4'; log=open(root+'/logs/full.log','w'); cmd=['python3','/content/scripts/train_v4.py','--data',root+'/data','--base-model','/content/drive/MyDrive/fable200m_v3/full/hf','--out',root+'/full/hf','--save-steps','500']; p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print('pid',p.pid)" | colab exec -s v4
```

Expected: 245,076 examples, 3,830 steps. At v3-full A100 throughput, estimate
13–16 minutes plus evaluation.

## Final evaluation

Generate the same 100 fresh controls for v3-full and v4:

```bash
echo "import subprocess; root='/content/drive/MyDrive'; subprocess.run(['python3','/content/scripts/generate_v4_comparison.py','--v3',root+'/fable200m_v3/full/hf','--v4',root+'/fable200m_v4/full/hf','--controls-file',root+'/fable200m_v4/data/eval_controls.json','--out',root+'/fable200m_v4/results/full_generations_100.json'],check=True)" | colab exec -s v4 --timeout 900
```

Then run deterministic evaluation and strict Gemma judging locally:

```bash
uv run python scripts/evaluate_v4_comparison.py \
  --input runs/v4/results/full_generations_100.json \
  --out runs/v4/results/full_metrics_100.json

set -a; source .env; set +a
uv run python scripts/judge_v4_comparison.py \
  --input runs/v4/results/full_generations_100.json \
  --out runs/v4/results/full_judged_20.json
```

Gemma 4 supports `minimal` (thinking off) or `high` (thinking on), not a `low`
level. Use `minimal` for the paired run; the stricter `v2-strict` rubric is the
main calibration change.

## Acceptance gate

- Exact character at least 75%.
- Exact moral at least 90%.
- Both exact at least 70%.
- Clean ending 100%.
- Strict paired judge prefers v4 overall; grammar and moral clarity do not fall.
- Inspect and record low/median/high sample stories before promoting v4.

Stop the VM only after Drive artifacts and logs are confirmed:

```bash
colab stop -s v4
```
