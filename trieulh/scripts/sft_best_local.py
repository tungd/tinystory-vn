"""SFT trainability test (RAFT): SFT 30M-p2 trên các bản best-of-N (judge chấm cao nhất).
Nếu mean-score held-out TĂNG -> headroom huấn luyện được (DPO null là do data, không phải capacity).
Format = pretraining: cond \n <|story|> story <|end|>, loss-mask conditioning.
"""
import argparse, json, math, torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
SEP,END,SEQ=("<|story|>","<|end|>",512)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="out/30M-p2"); ap.add_argument("--corpus",default="data/orpo/sft_best.jsonl")
    ap.add_argument("--out",default="out/30M-sft"); ap.add_argument("--epochs",type=float,default=3.0); ap.add_argument("--lr",type=float,default=1e-5)
    a=ap.parse_args()
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    tok=PreTrainedTokenizerFast(tokenizer_file=f"{a.model}/tokenizer.json",unk_token="<|unk|>",pad_token="<|pad|>",eos_token="<|end|>")
    rows=[json.loads(l) for l in open(a.corpus)]
    print(f"SFT trên {len(rows)} bản best-of-N",flush=True)
    def enc(prompt,story):
        cond=prompt+"\n"+SEP
        ids=tok(cond+story.strip()+END,truncation=True,max_length=SEQ)["input_ids"]
        nc=min(len(tok(cond)["input_ids"]),len(ids))
        return {"input_ids":ids,"labels":[-100]*nc+ids[nc:]}
    DS=[enc(r["prompt"],r["story"]) for r in rows]
    def coll(fs):
        m=max(len(f["input_ids"]) for f in fs); pad=tok.pad_token_id
        fill=lambda s,v:s+[v]*(m-len(s))
        return {"input_ids":torch.tensor([fill(f["input_ids"],pad) for f in fs]),
                "labels":torch.tensor([fill(f["labels"],-100) for f in fs]),
                "attention_mask":torch.tensor([[1]*len(f["input_ids"])+[0]*(m-len(f["input_ids"])) for f in fs])}
    model=AutoModelForCausalLM.from_pretrained(a.model).to(dev)
    args=TrainingArguments(output_dir=a.out,num_train_epochs=a.epochs,learning_rate=a.lr,
        per_device_train_batch_size=4,gradient_accumulation_steps=2,logging_steps=5,
        save_strategy="no",report_to=[],fp16=False,bf16=False)
    Trainer(model=model,args=args,train_dataset=DS,data_collator=coll).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    p=f"{a.out}/tokenizer_config.json"; c=json.load(open(p)); c["tokenizer_class"]="PreTrainedTokenizerFast"; json.dump(c,open(p,"w"))
    print(f"saved -> {a.out}",flush=True)
if __name__=="__main__": main()
