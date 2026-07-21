"""Reward model pairwise (Bradley-Terry): loss = -log sigmoid(r_chosen - r_rejected)
trên 194 cặp DPO (margin >= 1.0). Tín hiệu contrastive mạnh hơn pointwise MSE
(bản pointwise chỉ đạt Spearman 0.22 — collapse về predict mean).
Validate: pairwise accuracy trên cặp held-out + Spearman trên data/rm/val.jsonl.
"""
import json, math, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

SEP, END, SEQ = "<|story|>", "<|end|>", 512
dev = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, LR = 5, 1e-5

tok = PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",
                              unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")

class RewardModel(nn.Module):
    def __init__(self, path="out/30M-p2"):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(path).model
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1
        return self.head(h[torch.arange(h.size(0)), last]).squeeze(-1)

def enc(prompt, story):
    return tok(prompt + "\n" + SEP + story.strip() + END,
               truncation=True, max_length=SEQ)["input_ids"]

pairs = []
for l in open("data/orpo/pairs.jsonl"):
    r = json.loads(l)
    if "score_chosen" in r:
        pairs.append((enc(r["prompt"], r["chosen"]), enc(r["prompt"], r["rejected"])))
import random
random.Random(42).shuffle(pairs)
nval = 30
vpairs, tpairs = pairs[:nval], pairs[nval:]
print(f"pairs train {len(tpairs)} val {len(vpairs)}", flush=True)

val_rows = [json.loads(l) for l in open("data/rm/val.jsonl")]
val_pt = [(enc(r["prompt"], r["story"]), r["score"]) for r in val_rows]

def pad_batch(seqs):
    m = max(len(s) for s in seqs); p = tok.pad_token_id
    ids = torch.tensor([s + [p] * (m - len(s)) for s in seqs])
    att = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs])
    return ids.to(dev), att.to(dev)

def spearman(a, b):
    def rank(x):
        s = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
        for pos, i in enumerate(s): r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0

@torch.no_grad()
def validate(model):
    model.eval()
    ok = 0
    for c, r in vpairs:
        ids, att = pad_batch([c, r])
        s = model(ids, att)
        ok += int(s[0].item() > s[1].item())
    preds, ys = [], []
    for i in range(0, len(val_pt), 8):
        chunk = val_pt[i:i + 8]
        ids, att = pad_batch([x[0] for x in chunk])
        preds += model(ids, att).tolist(); ys += [x[1] for x in chunk]
    return ok / len(vpairs), spearman(preds, ys)

model = RewardModel().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
for ep in range(EPOCHS):
    model.train()
    random.Random(ep).shuffle(tpairs)
    tot = 0.0
    for j in range(0, len(tpairs), 2):  # 2 cặp = 4 sequence mỗi batch
        chunk = tpairs[j:j + 2]
        seqs = [s for c, r in chunk for s in (c, r)]
        ids, att = pad_batch(seqs)
        s = model(ids, att)
        diff = s[0::2] - s[1::2]
        loss = -torch.nn.functional.logsigmoid(diff).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot += loss.item() * len(chunk)
    acc, sp = validate(model)
    print(f"ep{ep+1} loss={tot/len(tpairs):.4f} val_pair_acc={acc:.3f} val_spearman={sp:.3f}", flush=True)
torch.save({"head": model.head.state_dict(), "backbone": model.backbone.state_dict()}, "out/rm-30m-bt.pt")
json.dump({"val_pair_acc": acc, "val_spearman": sp, "n_pairs_train": len(tpairs), "n_pairs_val": len(vpairs)},
          open("trieulh/report/data/rm_bt_metrics.json", "w"))
print("RM-BT DONE -> out/rm-30m-bt.pt", flush=True)
