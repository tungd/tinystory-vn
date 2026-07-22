"""Distillation bước 1: teacher Qwen3-4B sinh corpus truyện theo 5 slot.
Khác 4 thí nghiệm null trước: đây là tín hiệu OFF-DISTRIBUTION thật (không rút từ
phân bố của SLM). Ràng buộc phong cách TF1: tiếng Anh đơn giản (6-10 tuổi),
150-300 từ, kết bằng moral - để khớp vocab 12k BPE + Flesch band của SLM.
Lọc cứng: story phải <= MAX_TOK token theo tokenizer SLM (vừa context 512).
Resume-safe: append từng dòng, bỏ qua prompt đã xong. Output data/distill/corpus.jsonl.
"""
import json, os, requests
from transformers import PreTrainedTokenizerFast

TARGET = 600
MAX_TOK = 400          # token SLM cho story (prompt ~80-100 + story <= 400 < 512)
OUT = "data/distill/corpus.jsonl"
REJ = "data/distill/rejected.txt"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3-4b-instruct"

SYSTEM = (
    "You write children's fables for ages 6-10. Rules: use only simple, common English "
    "words (no rare or fancy vocabulary); STRICTLY 150-250 words; short sentences; a clear "
    "beginning, middle and end; finish with the moral as the final sentence. "
    "Follow the requested narrative elements exactly. Output ONLY the story text - "
    "no title, no preamble, no markdown."
)

os.makedirs("data/distill", exist_ok=True)
tok = PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",
                              unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")

def teacher(prompt):
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 500},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

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
    import re
    p_teacher = re.sub(r"about \d+-\d+ words", "about 170-240 words", p)
    try:
        story = teacher(p_teacher)
    except Exception as e:
        print(f"teacher err ({e}); thử lại prompt sau", flush=True)
        continue
    for a, b in [("\u2014", ", "), ("\u2013", "-"), ("\u2018", "'"), ("\u2019", "'"),
                 ("\u201c", '"'), ("\u201d", '"'), ("\u2026", "...")]:
        story = story.replace(a, b)
    n = len(tok(story)["input_ids"])
    words = len(story.split())
    if n <= MAX_TOK and 120 <= words <= 280 and "\n#" not in story and "**" not in story:
        f.write(json.dumps({"prompt": p, "story": story, "tok": n, "words": words,
                            "src": "qwen3-4b"}, ensure_ascii=False) + "\n"); f.flush()
        kept += 1
        print(f"{kept}/{TARGET} tok={n} words={words}", flush=True)
    else:
        rj.write(p + "\n"); rj.flush()
        print(f"{kept}/{TARGET} reject tok={n} words={words}", flush=True)
f.close(); rj.close()
print("DISTILL-GEN DONE" if kept >= TARGET else "PARTIAL", flush=True)
