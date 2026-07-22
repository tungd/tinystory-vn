"""RM bước 0: gom mọi (prompt, story, judge_score) đã chấm từ các thí nghiệm trước
làm data train reward model. Dedupe theo story. Output data/rm/scored.jsonl + split val.
Nguồn: pairs.jsonl (chosen + rejected), sft_best.jsonl, raft corpus, headroom (best_story).
"""
import json, os, random
os.makedirs("data/rm", exist_ok=True)
pool = {}
def add(prompt, story, score, src):
    key = story.strip()[:200]
    if key and key not in pool:
        pool[key] = {"prompt": prompt, "story": story.strip(), "score": float(score), "src": src}
for l in open("data/orpo/pairs.jsonl"):
    r = json.loads(l)
    if "score_chosen" in r:
        add(r["prompt"], r["chosen"], r["score_chosen"], "pair_chosen")
        add(r["prompt"], r["rejected"], r["score_rejected"], "pair_rejected")
for l in open("data/orpo/sft_best.jsonl"):
    r = json.loads(l)
    add(r["prompt"], r["story"], r["score"], "sft_best")
for l in open("data/raft/corpus.jsonl"):
    r = json.loads(l)
    add(r["prompt"], r["story"], r["score"], "raft")
if os.path.exists("data/orpo/headroom.jsonl"):
    tests = [json.loads(l) for l in open("data/tf1/test.jsonl")]
    for l in open("data/orpo/headroom.jsonl"):
        r = json.loads(l)
        cond = tests[r["i"]]["text"][:tests[r["i"]]["cond_len"]].rstrip("\n")
        add(cond, r["best_story"]["story"], r["best_story"]["overall"], "headroom")
rows = list(pool.values())
random.Random(42).shuffle(rows)
nval = max(30, len(rows) // 10)
with open("data/rm/val.jsonl", "w") as f:
    for r in rows[:nval]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/rm/train.jsonl", "w") as f:
    for r in rows[nval:]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
import statistics
print(f"total {len(rows)} (train {len(rows)-nval} / val {nval})")
print("src:", Counter(r["src"] for r in rows))
print("score mean/min/max:", round(statistics.mean(r['score'] for r in rows), 2),
      min(r['score'] for r in rows), max(r['score'] for r in rows))
