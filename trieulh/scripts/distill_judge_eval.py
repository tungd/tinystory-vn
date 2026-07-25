"""Đánh giá RAFT: judge eval trên 15 prompt held-out (data/tf1/test.jsonl[:15],
cùng protocol với dpo194_judge_eval) + guard perplexity held-out (test.jsonl[100:140]).
So sánh out/30M-p2 (baseline) vs out/30M-raft. Output trieulh/report/data/distill_judge_eval.json.
"""
import json, math, torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from app import judge as J

SEP, END = "<|story|>", "<|end|>"
dev = "mps"
NP = 45
rows = [json.loads(l) for l in open("data/tf1/test.jsonl")]
tests = rows[:NP]
ppl_rows = rows[100:140]

def load(path):
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json",
                                  unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")
    m = AutoModelForCausalLM.from_pretrained(path).to(dev).eval()
    return tok, m

def gen(tok, m, cond, seed):
    torch.manual_seed(seed)
    eos = tok.convert_tokens_to_ids(END)
    ids = tok(cond + "\n" + SEP, return_tensors="pt", return_token_type_ids=False).to(dev)
    with torch.no_grad():
        o = m.generate(**ids, max_new_tokens=440, do_sample=True, temperature=0.8, top_p=0.9,
                       repetition_penalty=1.1, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    t = tok.decode(o[0], skip_special_tokens=False)
    return t.split(SEP, 1)[1].replace(END, "").strip() if SEP in t else t

def heldout_ppl(tok, m):
    tot, cnt = 0.0, 0
    for r in ppl_rows:
        ids = tok(r["text"], return_tensors="pt", return_token_type_ids=False,
                  truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out = m(**ids, labels=ids["input_ids"])
        n_tok = ids["input_ids"].shape[1]
        tot += out.loss.item() * n_tok; cnt += n_tok
    return math.exp(tot / cnt)

res = {}
# Resume-safe: mỗi (model, prompt) ghi một dòng progress, chạy lại bỏ qua mục đã xong.
import os
PROG = "data/distill/eval_progress.jsonl"
done = {}
if os.path.exists(PROG):
    for l in open(PROG):
        r = json.loads(l)
        done[(r["model"], r["i"])] = r

pf = open(PROG, "a")
for name, path in [("p2", "out/30M-p2"), ("distill", "out/30M-distill")]:
    tok = m = None
    ov, adh = [], []
    for i, r in enumerate(tests):
        key = (name, i)
        if key in done:
            ov.append(done[key]["ov"]); adh.append(done[key]["adh"])
            continue
        if m is None:
            tok, m = load(path)
        cond = r["text"][:r["cond_len"]].rstrip("\n")
        story = gen(tok, m, cond, seed=1234 + i)
        e = J.evaluate(story, cond, model="qwen3-4b-instruct")
        row = {"model": name, "i": i, "ov": e["overall"], "adh": e["prompt_adherence"]}
        pf.write(json.dumps(row) + "\n"); pf.flush()
        ov.append(row["ov"]); adh.append(row["adh"])
        print(f"{name} {i+1}/{NP} ov={row['ov']:.2f} adh={row['adh']}", flush=True)
    if m is None:
        tok, m = load(path)
    ppl = heldout_ppl(tok, m)
    res[name] = {"ov": ov, "adh": adh, "ppl": ppl}
    print(f"{name}: mean_ov={sum(ov)/len(ov):.3f} mean_adh={sum(adh)/len(adh):.2f} ppl={ppl:.3f}", flush=True)
    del m; torch.mps.empty_cache()
pf.close()

json.dump(res, open("trieulh/report/data/distill_judge_eval.json", "w"), ensure_ascii=False)
print("RAFT-EVAL DONE", flush=True)
