"""Sinh corpus SFT best-of-N: mỗi prompt sinh K bản, judge chọn bản overall cao nhất.
Prompt lấy từ tập ORPO (offset 500, KHÁC tập eval test.jsonl[:15] -> không leak).
Resume-safe: append từng prompt, bỏ qua prompt đã xong. Output data/orpo/sft_best.jsonl.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from app import judge as J
SEP,END="<|story|>","<|end|>"; dev="mps"; K=3; TARGET=50
OUT="data/orpo/sft_best.jsonl"
tok=PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",unk_token="<|unk|>",pad_token="<|pad|>",eos_token="<|end|>")
eos=tok.convert_tokens_to_ids(END)
m=AutoModelForCausalLM.from_pretrained("out/30M-p2").to(dev).eval()
def gen(cond,temp):
    ids=tok(cond+"\n"+SEP,return_tensors="pt",return_token_type_ids=False).to(dev)
    with torch.no_grad():
        o=m.generate(**ids,max_new_tokens=440,do_sample=True,temperature=temp,top_p=0.9,repetition_penalty=1.1,eos_token_id=eos,pad_token_id=tok.pad_token_id)
    t=tok.decode(o[0],skip_special_tokens=False); return t.split(SEP,1)[1].replace(END,"").strip() if SEP in t else t
prompts=[json.loads(l)["prompt"] for l in open("data/orpo/prompts.jsonl")]
done=set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["prompt"])
kept=len(done)
f=open(OUT,"a")
temps=[0.5,0.8,1.1]
for p in prompts:
    if kept>=TARGET: break
    if p in done: continue
    scored=[(J.evaluate(s,p,model="qwen3-4b-instruct")["overall"], s) for s in (gen(p,t) for t in temps)]
    best=max(scored,key=lambda x:x[0])
    f.write(json.dumps({"prompt":p,"story":best[1],"score":best[0]},ensure_ascii=False)+"\n"); f.flush()
    kept+=1; print(f"{kept}/{TARGET} best={best[0]:.2f}",flush=True)
f.close(); print("SFT-BEST DONE" if kept>=TARGET else "PARTIAL",flush=True)
