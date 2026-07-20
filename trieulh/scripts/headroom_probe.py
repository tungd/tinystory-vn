import json, os, torch, statistics, sys
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from app import judge as J
SEP,END="<|story|>","<|end|>"; dev="mps"
OUT="data/orpo/headroom.jsonl"; TEMPS=[0.5,0.8,1.1]; NP=10
tok=PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",unk_token="<|unk|>",pad_token="<|pad|>",eos_token="<|end|>")
eos=tok.convert_tokens_to_ids(END)
done=set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["i"])
m=AutoModelForCausalLM.from_pretrained("out/30M-p2").to(dev).eval()
def gen(cond,temp):
    ids=tok(cond+"\n"+SEP,return_tensors="pt",return_token_type_ids=False).to(dev)
    with torch.no_grad():
        o=m.generate(**ids,max_new_tokens=440,do_sample=True,temperature=temp,top_p=0.9,repetition_penalty=1.1,eos_token_id=eos,pad_token_id=tok.pad_token_id)
    t=tok.decode(o[0],skip_special_tokens=False); return t.split(SEP,1)[1].replace(END,"").strip() if SEP in t else t
tests=[json.loads(l) for l in open("data/tf1/test.jsonl")][:NP]
torch.manual_seed(11)
f=open(OUT,"a")
for i,t in enumerate(tests):
    if i in done: continue
    cond=t["text"][:t["cond_len"]].rstrip("\n"); scored=[]
    for tp in TEMPS:
        s=gen(cond,tp); sc=J.evaluate(s,cond,model="qwen3-4b-instruct")
        scored.append({"temp":tp,"overall":sc["overall"],"adh":sc["prompt_adherence"],"story":s})
    ov=[x["overall"] for x in scored]
    rec={"i":i,"mean":statistics.mean(ov),"best":max(ov),"best_story":max(scored,key=lambda x:x["overall"])}
    f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
    print(f"{i+1}/{NP}: mean {rec['mean']:.2f} best {rec['best']:.2f}",flush=True)
f.close(); print("PROBE DONE",flush=True)
