"""GRPO-lite = REINFORCE + baseline theo nhóm (Week10 tr.14-19, variance reduction).
Khác DPO/SFT/RAFT (exploitation trên data tĩnh): on-policy rollout MỚI mỗi step,
advantage = (r_i - mean(nhóm)) / std(nhóm) -> có gradient ÂM đẩy xuống sample dưới baseline.

loss = -mean_i( adv_i * mean_t log pi(y_t | y<t, x) ) + beta * KL(pi || pi_ref)

Reward backend: --reward judge (qwen3-4b-instruct, chậm ~15s/call) | rm (out/rm-30m-bt.pt, nhanh).
Guard: perplexity held-out mỗi EVAL_EVERY step, dừng nếu drift > +10%.
Resume-safe: checkpoint + step vào out/30M-grpo/state.json.
"""
import argparse, json, math, os, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

SEP, END, SEQ = "<|story|>", "<|end|>", 512
dev = "mps" if torch.backends.mps.is_available() else "cpu"

ap = argparse.ArgumentParser()
ap.add_argument("--reward", choices=["judge", "rm"], default="judge")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--batch-prompts", type=int, default=4)
ap.add_argument("--group", type=int, default=4)      # G rollout / prompt
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--beta-kl", type=float, default=0.05)
ap.add_argument("--out", default="out/30M-grpo")
a = ap.parse_args()

tok = PreTrainedTokenizerFast(tokenizer_file="out/30M-p2/tokenizer.json",
                              unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")
EOS = tok.convert_tokens_to_ids(END)

# --- policy + ref (resume nếu có checkpoint) ---
start_step = 0
src = "out/30M-p2"
state_p = f"{a.out}/state.json"
if os.path.exists(state_p):
    start_step = json.load(open(state_p))["step"]
    src = a.out
    print(f"resume từ step {start_step}", flush=True)
policy = AutoModelForCausalLM.from_pretrained(src).to(dev)
ref = AutoModelForCausalLM.from_pretrained("out/30M-p2").to(dev).eval()
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)

# --- reward backend ---
if a.reward == "judge":
    from app import judge as J
    def reward_fn(prompt, story):
        try:
            return J.evaluate(story, prompt, model="qwen3-4b-instruct")["overall"]
        except Exception as e:
            print(f"judge err: {e}", flush=True)
            return None
else:
    class RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModelForCausalLM.from_pretrained("out/30M-p2").model
            self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        def forward(self, input_ids, attention_mask):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            last = attention_mask.sum(1) - 1
            return self.head(h[torch.arange(h.size(0)), last]).squeeze(-1)
    rm = RewardModel().to(dev).eval()
    sd = torch.load("out/rm-30m-bt.pt", map_location=dev, weights_only=True)
    rm.backbone.load_state_dict(sd["backbone"]); rm.head.load_state_dict(sd["head"])
    @torch.no_grad()
    def reward_fn(prompt, story):
        ids = tok(prompt + "\n" + SEP + story + END, truncation=True, max_length=SEQ,
                  return_tensors="pt", return_token_type_ids=False).to(dev)
        return rm(ids["input_ids"], ids["attention_mask"]).item()

# --- data ---
prompts = [json.loads(l)["prompt"] for l in open("data/orpo/prompts.jsonl")]
ppl_rows = [json.loads(l)["text"] for l in open("data/tf1/test.jsonl")][100:130]

@torch.no_grad()
def heldout_ppl(m):
    tot, cnt = 0.0, 0
    for txt in ppl_rows:
        ids = tok(txt, return_tensors="pt", return_token_type_ids=False,
                  truncation=True, max_length=SEQ).to(dev)
        out = m(**ids, labels=ids["input_ids"])
        n = ids["input_ids"].shape[1]; tot += out.loss.item() * n; cnt += n
    return math.exp(tot / cnt)

@torch.no_grad()
def rollout(prompt, n):
    cond = prompt + "\n" + SEP
    ids = tok(cond, return_tensors="pt", return_token_type_ids=False).to(dev)
    outs = policy.generate(**ids, max_new_tokens=380, do_sample=True, temperature=0.9,
                           top_p=0.95, num_return_sequences=n,
                           eos_token_id=EOS, pad_token_id=tok.pad_token_id)
    cond_len = ids["input_ids"].shape[1]
    stories, seqs = [], []
    for o in outs:
        t = tok.decode(o, skip_special_tokens=False)
        s = t.split(SEP, 1)[1].replace(END, "").strip() if SEP in t else ""
        # cắt padding sau EOS
        toks = o.tolist()
        if EOS in toks[cond_len:]:
            toks = toks[:cond_len + toks[cond_len:].index(EOS) + 1]
        stories.append(s); seqs.append((toks, cond_len))
    return stories, seqs

def logprobs_mean(model, toks, cond_len, need_grad):
    ids = torch.tensor([toks]).to(dev)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        logits = model(ids).logits[0, :-1]
        lp = torch.log_softmax(logits.float(), -1)
        tgt = ids[0, 1:]
        tok_lp = lp[torch.arange(len(tgt)), tgt]
        story_lp = tok_lp[cond_len - 1:]  # chỉ tính phần story
    return story_lp.mean(), story_lp

ppl0 = heldout_ppl(policy)
print(f"reward={a.reward} steps={a.steps} B={a.batch_prompts} G={a.group} lr={a.lr} "
      f"beta_kl={a.beta_kl} | ppl0={ppl0:.3f}", flush=True)
os.makedirs(a.out, exist_ok=True)
LOG = f"{a.out}/grpo_log.jsonl"
rng = random.Random(1000 + start_step)
EVAL_EVERY = 10

for step in range(start_step, a.steps):
    batch = rng.sample(prompts, a.batch_prompts)
    losses, rmeans, rstds, kls = [], [], [], []
    opt.zero_grad()
    n_items = 0
    for prompt in batch:
        stories, seqs = rollout(prompt, a.group)
        rewards = []
        keep = []
        for s, sq in zip(stories, seqs):
            r = reward_fn(prompt, s) if s else None
            if r is not None:
                rewards.append(r); keep.append(sq)
        if len(rewards) < 2:
            continue
        t_r = torch.tensor(rewards)
        adv = (t_r - t_r.mean()) / (t_r.std() + 1e-4)
        rmeans.append(t_r.mean().item()); rstds.append(t_r.std().item())
        for (toks, cl), ad in zip(keep, adv.tolist()):
            mean_lp, story_lp = logprobs_mean(policy, toks, cl, need_grad=True)
            with torch.no_grad():
                _, ref_lp = logprobs_mean(ref, toks, cl, need_grad=False)
            kl = (story_lp - ref_lp).mean()  # xấp xỉ KL trên trajectory
            loss = -ad * mean_lp + a.beta_kl * kl
            (loss / (a.batch_prompts * a.group)).backward()
            losses.append(loss.item()); kls.append(kl.item()); n_items += 1
    if n_items == 0:
        print(f"step {step+1}: không có nhóm hợp lệ, bỏ qua", flush=True)
        continue
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    row = {"step": step + 1, "loss": sum(losses) / len(losses),
           "reward_mean": sum(rmeans) / len(rmeans), "reward_std": sum(rstds) / len(rstds),
           "kl": sum(kls) / len(kls), "n": n_items}
    if (step + 1) % EVAL_EVERY == 0 or step + 1 == a.steps:
        row["ppl"] = heldout_ppl(policy)
        if row["ppl"] > ppl0 * 1.10:
            print(f"DỪNG: ppl drift {row['ppl']:.3f} > {ppl0*1.1:.3f}", flush=True)
            break
    open(LOG, "a").write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    policy.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump({"step": step + 1}, open(state_p, "w"))
print("GRPO DONE", flush=True)
