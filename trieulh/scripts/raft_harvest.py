"""RAFT bước 0: thu hoạch truyện đã judge >= NGƯỠNG từ các thí nghiệm trước.
Nguồn: sft_best.jsonl (best-of-3) + pairs.jsonl (chosen của DPO). Dedupe theo prompt.
Output: data/raft/corpus.jsonl (schema {prompt, story, score, src}).
"""
import json, os
THRESH = 9.0
OUT = "data/raft/corpus.jsonl"
os.makedirs("data/raft", exist_ok=True)
pool = {}
for l in open("data/orpo/sft_best.jsonl"):
    r = json.loads(l)
    if r["score"] >= THRESH:
        pool[r["prompt"]] = {"prompt": r["prompt"], "story": r["story"], "score": r["score"], "src": "sft_best"}
for l in open("data/orpo/pairs.jsonl"):
    r = json.loads(l)
    if "score_chosen" in r and float(r["score_chosen"]) >= THRESH and r["prompt"] not in pool:
        pool[r["prompt"]] = {"prompt": r["prompt"], "story": r["chosen"], "score": float(r["score_chosen"]), "src": "dpo_chosen"}
with open(OUT, "w") as f:
    for row in pool.values():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"harvested {len(pool)} stories >= {THRESH} -> {OUT}", flush=True)
