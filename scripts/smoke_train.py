import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, ".")
from tokenizers import Tokenizer
from transformers import (
    PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)
from datasets import Dataset

DATA = Path("data/fable200m_smoke")
tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok)
hf_tok.pad_token = "</story>"
hf_tok.eos_token = "</story>"

texts = [json.loads(l) for l in open(DATA / "fables.jsonl") if l.strip()]
ds = Dataset.from_dict({"text": texts})
def encode(ex):
    ids = tok.encode(ex["text"]).ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}
ds = ds.map(encode, remove_columns=["text"]).train_test_split(test_size=0.05)

cfg = GPT2Config(vocab_size=tok.get_vocab_size(), n_positions=512,
                 n_embd=256, n_layer=6, n_head=8,
                 resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                 bos_token_id=tok.get_vocab()["<story>"],
                 eos_token_id=tok.get_vocab()["</story>"])
model = GPT2LMHeadModel(cfg).to("mps")
print("params:", sum(p.numel() for p in model.parameters()))

collator = DataCollatorForLanguageModeling(tokenizer=hf_tok, mlm=False)
args = TrainingArguments(
    output_dir="data/fable200m_smoke/ckpt", per_device_train_batch_size=8,
    gradient_accumulation_steps=2, max_steps=200, learning_rate=5e-4,
    warmup_steps=20, lr_scheduler_type="cosine", logging_steps=20,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["test"], data_collator=collator)
trainer.train()
trainer.save_model("data/fable200m_smoke/ckpt")
hf_tok.save_pretrained("data/fable200m_smoke/ckpt")

# ---- generate ----
gen = torch.nn.functional
from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=hf_tok, device="mps")
PREFIX = "<char> {character} </char>\n<moral> {moral} </moral>\n<story>\n"
prompt = PREFIX.format(character="a clever fox", moral="cleverness beats brute force")
out = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.9, top_p=0.9, repetition_penalty=1.3)
print("\n=== GENERATED ===\n", out[0]["generated_text"])
