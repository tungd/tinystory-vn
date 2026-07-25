"""Reward model (Week10 Actor-Critic, tr.20-21): backbone 30M-p2 + linear head trên
hidden state token cuối, hồi quy judge score (chuẩn hóa /10, MSE).
Mục đích: reward NHANH + ít nhiễu hơn gọi judge Qwen-4B cho GRPO-lite và rerank best-of-N.
Validate: Pearson/Spearman trên val held-out — nếu Spearman < 0.5 thì RM chưa đủ tốt để rank.
"""
import json, math, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

SEP, END, SEQ = "<|story|>", "<|end|>", 512
dev = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, LR, BS = 4, 1e-5, 8

tok = PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",
                              unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")

class RewardModel(nn.Module):
    def __init__(self, path="out/30M-p2"):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(path).model  # bỏ lm_head
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1  # vị trí token thật cuối cùng
        pooled = h[torch.arange(h.size(0)), last]
        return self.head(pooled).squeeze(-1)

def encode(rows):
    out = []
    for r in rows:
        ids = tok(r["prompt"] + "\n" + SEP + r["story"] + END,
                  truncation=True, max_length=SEQ)["input_ids"]
        out.append((ids, r["score"] / 10.0))
    return out

def batches(data, bs, shuffle=True):
    idx = torch.randperm(len(data)).tolist() if shuffle else range(len(data))
    buf = []
    for i in idx:
        buf.append(data[i])
        if len(buf) == bs:
            yield buf; buf = []
    if buf:
        yield buf

def collate(buf):
    m = max(len(x[0]) for x in buf); pad = tok.pad_token_id
    ids = torch.tensor([x[0] + [pad] * (m - len(x[0])) for x in buf])
    att = torch.tensor([[1] * len(x[0]) + [0] * (m - len(x[0])) for x in buf])
    y = torch.tensor([x[1] for x in buf])
    return ids.to(dev), att.to(dev), y.to(dev)

def spearman(a, b):
    def rank(x):
        s = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(s):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0

def evaluate(model, data):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for buf in batches(data, BS, shuffle=False):
            ids, att, y = collate(buf)
            p = model(ids, att)
            preds += p.tolist(); ys += y.tolist()
    mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)
    mp, my = sum(preds) / len(preds), sum(ys) / len(ys)
    num = sum((p - mp) * (y - my) for p, y in zip(preds, ys))
    den = math.sqrt(sum((p - mp) ** 2 for p in preds) * sum((y - my) ** 2 for y in ys))
    pear = num / den if den else 0.0
    return mse, pear, spearman(preds, ys)

train = encode([json.loads(l) for l in open("data/rm/train.jsonl")])
val = encode([json.loads(l) for l in open("data/rm/val.jsonl")])
model = RewardModel().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lossf = nn.MSELoss()
print(f"train {len(train)} val {len(val)}", flush=True)
for ep in range(EPOCHS):
    model.train()
    tot, n = 0.0, 0
    for buf in batches(train, BS):
        ids, att, y = collate(buf)
        opt.zero_grad()
        loss = lossf(model(ids, att), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(buf); n += len(buf)
    mse, pear, sp = evaluate(model, val)
    print(f"ep{ep+1} train_mse={tot/n:.4f} val_mse={mse:.4f} pearson={pear:.3f} spearman={sp:.3f}", flush=True)
torch.save({"head": model.head.state_dict(), "backbone": model.backbone.state_dict()}, "out/rm-30m.pt")
json.dump({"val_mse": mse, "pearson": pear, "spearman": sp, "n_train": len(train), "n_val": len(val)},
          open("trieulh/report/data/rm_metrics.json", "w"))
print("RM DONE -> out/rm-30m.pt", flush=True)
