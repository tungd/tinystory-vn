# Failure-Focused LoRA Plan

This is the training contribution for the project. It is intentionally different
from the two reference projects:

- not from-scratch SLM training
- not DPO / ORPO preference alignment
- not broad synthetic SFT

The idea is to train a small LoRA only on failures discovered by the evaluation
taxonomy.

## Research Question

Can a small failure-focused LoRA improve fable reliability more efficiently than
broad SFT?

## Hypothesis

Broad SFT on imperfect synthetic stories can damage fluency. A targeted LoRA
trained only on corrected failure cases should improve moral reliability and
ending completeness while preserving more of the base model's fluency.

## Dataset Format

Each row contains:

- original prompt slots
- detected failure reasons
- corrected story from the repair pipeline
- Llama 3.2 chat-formatted `text` field for SFT

Current seed dataset:

```text
data/failure_focused_lora/
data/failure_focused_lora.zip
```

This seed is built from the 25-prompt benchmark. It is useful for verifying the
pipeline, but it is too small for a serious LoRA run. Extend it to 100-300 rows
before training.

Extended dataset:

```text
data/failure_focused_lora_300/
data/failure_focused_lora_300.zip
```

This combines observed failure rows with TF1 corrected targets tagged by the
same failure taxonomy. Use the extended zip for Kaggle training.

## Local Build Command

```powershell
.\.venv\Scripts\python.exe scripts\build_failure_focused_lora.py
.\.venv\Scripts\python.exe scripts\extend_failure_focused_lora.py
```

## How To Extend To 100-300 Examples

1. Create more prompts with the same schema as `data/test_prompts.jsonl`.
2. Generate base outputs for those prompts.
3. Run `scripts/evaluate_outputs.py`.
4. Run `scripts/run_enhanced_base.py --rewrite`.
5. Run `scripts/build_failure_focused_lora.py` using those new files.

The key point is that only failed or corrected cases enter the LoRA dataset.

## Kaggle Training Steps

Upload `data/failure_focused_lora_300.zip` to Kaggle, then run:

```python
import zipfile, os

zip_path = "/kaggle/input/failure-focused-lora-300/failure_focused_lora_300.zip"
out_dir = "/kaggle/working/failure_focused_lora_300"
os.makedirs(out_dir, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
```

Install dependencies:

```python
!pip install -q unsloth transformers datasets trl accelerate bitsandbytes peft
```

Load dataset:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={
        "train": "/kaggle/working/failure_focused_lora_300/train.jsonl",
        "validation": "/kaggle/working/failure_focused_lora_300/valid.jsonl",
    },
)
print(ds)
print(ds["train"][0]["text"][:1000])
```

Train LoRA with Unsloth:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=5410,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        output_dir="/kaggle/working/failure_lora_out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        seed=5410,
    ),
)

trainer.train()
model.save_pretrained("/kaggle/working/failure_lora_adapter")
tokenizer.save_pretrained("/kaggle/working/failure_lora_adapter")
```

Export merged GGUF:

```python
model.save_pretrained_gguf(
    "/kaggle/working/failure_lora_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

Download the `.gguf`, put it in `experiments/`, create an Ollama Modelfile, and
evaluate it against:

- Base
- SFT Clean 3K
- Base + Postprocess
- Base + Repair
- Failure-focused LoRA

## Expected Contribution

The final report should not claim LoRA is automatically better. The contribution
is the comparison:

1. broad SFT can overfit bad stylistic patterns;
2. postprocess fixes deterministic format errors cheaply;
3. repair handles selected severe cases;
4. failure-focused LoRA tests whether those corrections can be learned by the
   model with much less data than broad SFT.
