"""So sánh trực diện slm-30m-p2 vs slm-30m-dpo qua chính app API (localhost:8000).
Hai use case, cùng seed bắt cặp giữa 2 model:
  UC1: KHÔNG slot (free), length short, best_of_n=1
  UC2: ĐỦ 5 slot,          length short, best_of_n=1
Thu: story + observability meta (từ SSE) + quick evaluation (/evaluate, judge qwen3-4b).
Output: trieulh/report/data/p2_vs_dpo_headtohead.json (story đầy đủ để Claude chấm tay).
"""
import json, re, requests

API = "http://localhost:8000"
MODELS = ["slm-30m", "slm-30m-p2", "slm-30m-dpo", "slm-60m"]
SLOT_SETS = [
    {"character": "a shy hedgehog", "setting": "a moonlit orchard",
     "challenge": "afraid to ask for help", "outcome": "friends help when asked",
     "teaching": "asking for help is brave"},
    {"character": "a curious duckling", "setting": "a busy riverbank",
     "challenge": "gets lost following a butterfly", "outcome": "finds the way home by listening",
     "teaching": "listen before you leap"},
    {"character": "a proud rooster", "setting": "a small farm",
     "challenge": "refuses to share the best perch", "outcome": "learns sharing brings friends",
     "teaching": "sharing makes everyone happier"},
    {"character": "a patient turtle", "setting": "a sunny meadow",
     "challenge": "teased for being slow", "outcome": "wins respect by finishing the job",
     "teaching": "steady effort pays off"},
]
EMPTY = {"character": "", "setting": "", "challenge": "", "outcome": "", "teaching": ""}


def generate(model_id, slots, seed):
    body = {**slots, "length": "short", "model_id": model_id,
            "guardrail_enabled": False, "best_of_n": 1, "seed": seed}
    r = requests.post(f"{API}/generate/stream", json=body, stream=True, timeout=180)
    r.raise_for_status()
    story, meta = None, None
    for line in r.iter_lines(decode_unicode=True):
        if not line.startswith("data: "):
            continue
        d = json.loads(line[6:])
        if d.get("type") == "story":
            story = d.get("story") or d.get("text")
        elif d.get("type") == "meta":
            meta = d.get("meta") or {k: v for k, v in d.items() if k != "type"}
        elif d.get("type") == "done":
            story = story or d.get("story")
            meta = meta or d.get("meta")
    return story, meta


def evaluate(story, prompt):
    r = requests.post(f"{API}/evaluate", json={"story": story, "prompt": prompt}, timeout=300)
    r.raise_for_status()
    return r.json()


import os
OUT = "trieulh/report/data/p2_vs_dpo_headtohead.json"
results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(r["uc"], r["i"], r["model"]) for r in results}
for uc, slots_list in [("UC1_free", [EMPTY] * 4), ("UC2_slots", SLOT_SETS)]:
    for i, slots in enumerate(slots_list):
        seed = 7000 + i  # cùng seed cho cả 2 model -> so sánh bắt cặp
        for m in MODELS:
            if (uc, i, m) in done:
                continue
            story, meta = generate(m, slots, seed)
            assert story, f"no story {m} {uc} {i}"
            prompt_sent = (meta or {}).get("prompt_sent", "")
            ev = evaluate(story, prompt_sent)
            row = {"uc": uc, "i": i, "model": m, "seed": seed, "slots": slots,
                   "story": story, "meta": meta, "eval": ev}
            results.append(row)
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
            ov = ev.get("overall")
            print(f"{uc} #{i} {m}: overall={ov} words={len(story.split())}", flush=True)

print("HEAD2HEAD DONE:", len(results), "rows", flush=True)
