"""RAFT bước 1: sinh bổ sung corpus tới TARGET truyện đạt ngưỡng tuyệt đối >= 9.0.
Khác SFT-on-best cũ (giữ best của batch bất kể điểm): ở đây CHỈ nhận truyện >= THRESH.
Judge từng candidate, chấp nhận sớm khi đạt ngưỡng (tiết kiệm judge call).
Resume-safe: bỏ qua prompt đã có trong corpus HOẶC đã thử thất bại (reject log).
"""
import json, os, torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from app import judge as J

SEP, END = "<|story|>", "<|end|>"
dev = "mps"
THRESH = 9.0
TARGET = 200
OUT = "data/raft/corpus.jsonl"
REJ = "data/raft/rejected_prompts.txt"
TEMPS = [0.5, 0.8, 1.1]

tok = PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",
                              unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")
eos = tok.convert_tokens_to_ids(END)
m = AutoModelForCausalLM.from_pretrained("out/30M-p2").to(dev).eval()

def gen(cond, temp):
    ids = tok(cond + "\n" + SEP, return_tensors="pt", return_token_type_ids=False).to(dev)
    with torch.no_grad():
        o = m.generate(**ids, max_new_tokens=440, do_sample=True, temperature=temp, top_p=0.9,
                       repetition_penalty=1.1, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    t = tok.decode(o[0], skip_special_tokens=False)
    return t.split(SEP, 1)[1].replace(END, "").strip() if SEP in t else t

prompts = [json.loads(l)["prompt"] for l in open("data/orpo/prompts.jsonl")]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        done.add(json.loads(l)["prompt"])
rejected = set()
if os.path.exists(REJ):
    rejected = {l.rstrip("\n") for l in open(REJ)}
kept = len(done)
f = open(OUT, "a"); rj = open(REJ, "a")
for p in prompts:
    if kept >= TARGET:
        break
    if p in done or p in rejected:
        continue
    best = (0.0, None)
    for t in TEMPS:
        s = gen(p, t)
        sc = J.evaluate(s, p, model="qwen3-4b-instruct")["overall"]
        if sc > best[0]:
            best = (sc, s)
        if sc >= THRESH:
            break  # chấp nhận sớm: đã đạt ngưỡng tuyệt đối
    if best[0] >= THRESH:
        f.write(json.dumps({"prompt": p, "story": best[1], "score": best[0], "src": "raft_gen"},
                           ensure_ascii=False) + "\n"); f.flush()
        kept += 1
        print(f"{kept}/{TARGET} accept={best[0]:.2f}", flush=True)
    else:
        rj.write(p + "\n"); rj.flush()
        print(f"{kept}/{TARGET} reject best={best[0]:.2f}", flush=True)
f.close(); rj.close()
print("RAFT-GEN DONE" if kept >= TARGET else "PARTIAL", flush=True)
