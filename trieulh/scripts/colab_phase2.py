"""Phase-2 runner cho SLM 30M, chạy headless trên Colab VM qua `colab exec`.

Pipeline: định vị folder Drive (chứa ckpt_30M) -> clone repo -> build corpus v2
(cap "wise old owl" 10% + slot-dropout teaching/outcome 0.15, GIỮ tokenizer cũ)
-> encode -> resume train từ checkpoint pha 1 lên STEPS=3600 -> lưu 30M-p2 ->
analysis nhanh (perplexity + owl-rate) -> export GGUF p2. Artifact đặt hậu tố
p2, KHÔNG ghi đè kết quả Run 3. Mọi print đều flush để CLI stream được log.

Spec: docs/superpowers/specs/2026-07-11-slm-data-fix-phase2-design.md
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import deque

STEPS = 3600          # pha 1 dừng ở 1800 (checkpoint 1500); pha 2 chạy tiếp tới 3600
TRAIN_N = 400_000
SEQ_LEN = 512
BATCH_SIZE, GRAD_ACCUM = 32, 4
PEAK_LR, ADAM_BETAS, WEIGHT_DECAY, GRAD_CLIP = 3e-3, (0.9, 0.95), 0.1, 1.0
WARMUP_FRAC, DECAY_FRAC = 0.02, 0.20
LOG_EVERY = 25
SEP, END = "<|story|>", "<|end|>"


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(f"FAILED: {cmd}")


def find_drive_root():
    """Tìm thư mục Drive chứa ckpt_30M (BFS giới hạn độ sâu 3 cho nhanh)."""
    base = "/content/drive/MyDrive"
    q = deque([(base, 0)])
    while q:
        d, depth = q.popleft()
        try:
            entries = [e for e in os.scandir(d) if e.is_dir()]
        except (PermissionError, FileNotFoundError):
            continue
        for e in entries:
            if e.name == "ckpt_30M":
                return d
            if depth < 3 and not e.name.startswith("."):
                q.append((e.path, depth + 1))
    return None


def main():
    import torch
    assert torch.cuda.is_available(), "session khong co GPU"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    DRIVE = find_drive_root()
    assert DRIVE, "khong tim thay thu muc Drive chua ckpt_30M (da drivemount chua?)"
    print("DRIVE =", DRIVE, flush=True)

    # ── repo + deps ────────────────────────────────────────────────────────────
    if not os.path.exists("/content/tinystory-vn"):
        sh("git clone -q https://github.com/tungd/tinystory-vn.git /content/tinystory-vn")
    os.chdir("/content/tinystory-vn")
    sh("git fetch -q origin && git checkout -q feat/slm-pretrain-tf1 && git pull -q")
    sh('pip -q install "datasets>=2.20" "tokenizers>=0.19" "transformers>=4.44" accelerate textstat')

    # ── corpus v2 (cap owl + dropout rebalance), tokenizer GIỮ NGUYÊN ─────────
    cache = f"{DRIVE}/data_tf1_v2"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(f"{cache}/tokenizer.json"):
        print("khôi phục corpus v2 từ cache Drive...", flush=True)
        shutil.rmtree("data/tf1", ignore_errors=True)
        shutil.copytree(cache, "data/tf1")
    else:
        sh(f"python -m trieulh.scripts.prepare_tf1_pretrain --train-n {TRAIN_N} --test-n 500 "
           f"--min-words 60 --max-words 320 "
           f'--cap-phrase "wise old owl" --cap-frac 0.10 '
           f"--slot-dropout teaching=0.15 outcome=0.15 --out data/tf1")
        shutil.copy(f"{DRIVE}/data_tf1/tokenizer.json", "data/tf1/tokenizer.json")
        shutil.copytree("data/tf1", cache, dirs_exist_ok=True)
        print("corpus v2 đã build và cache lên Drive", flush=True)

    # kiểm chứng can thiệp data
    texts = [json.loads(l)["text"].lower() for l in open("data/tf1/train.jsonl")]
    owl = sum(1 for t in texts if "wise old owl" in t)
    teach = sum(1 for t in texts if "teaching/moral" in t)
    n = len(texts)
    print(f"corpus v2: {n} rows | owl {owl/n:.1%} (target ~10%) | "
          f"teaching-in-cond {teach/n:.1%} (target ~85%)", flush=True)

    # ── dataset + collator (như notebook) ─────────────────────────────────────
    import torch
    from transformers import (LlamaForCausalLM, PreTrainedTokenizerFast,
                              Trainer, TrainingArguments)

    tok = PreTrainedTokenizerFast(tokenizer_file="data/tf1/tokenizer.json",
            unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")

    def encode(row):
        ids = tok(row["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
        n_cond = min(len(tok(row["text"][:row["cond_len"]])["input_ids"]), len(ids))
        return {"input_ids": ids, "labels": [-100] * n_cond + ids[n_cond:]}

    print("encoding...", flush=True)
    DS = [encode(json.loads(l)) for _, l in zip(range(TRAIN_N), open("data/tf1/train.jsonl"))]
    print("encoded", len(DS), flush=True)

    def collator(features):
        pad = tok.pad_token_id
        m = max(len(f["input_ids"]) for f in features)
        fill = lambda seq, val: seq + [val] * (m - len(seq))
        return {
            "input_ids":      torch.tensor([fill(f["input_ids"], pad) for f in features]),
            "labels":         torch.tensor([fill(f["labels"], -100)   for f in features]),
            "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in features]),
        }

    # ── model: nạp checkpoint pha 1, train tiếp tới STEPS ─────────────────────
    ckpt_p2 = f"{DRIVE}/ckpt_30M_p2"
    p2 = sorted(glob.glob(f"{ckpt_p2}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    p1 = sorted(glob.glob(f"{DRIVE}/ckpt_30M/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume_from = p2[-1] if p2 else (p1[-1] if p1 else None)
    assert resume_from, "khong co checkpoint pha 1 de resume"
    print("resume từ:", resume_from, flush=True)

    model = LlamaForCausalLM.from_pretrained(resume_from)
    print("params(M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                  betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)

    def wsd(step):
        warm, dec = int(WARMUP_FRAC * STEPS), int(DECAY_FRAC * STEPS)
        if step < warm:
            return step / max(1, warm)
        if step > STEPS - dec:
            return max(0.0, (STEPS - step) / max(1, dec))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd)

    args = TrainingArguments(
        output_dir=ckpt_p2, max_steps=STEPS, fp16=True,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=GRAD_CLIP, logging_steps=LOG_EVERY,
        save_strategy="steps", save_steps=500, save_total_limit=2,
        lr_scheduler_type="constant", report_to=[],
        ignore_data_skip=True,   # corpus MỚI: bắt đầu từ đầu data, không skip 192k example
        disable_tqdm=True,       # log dòng thuần cho CLI stream
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS, data_collator=collator,
                      optimizers=(optimizer, scheduler))
    trainer.train(resume_from_checkpoint=resume_from)

    # ── lưu model p2 (không đè Run 3) ──────────────────────────────────────────
    for d in ("out/30M-p2", f"{DRIVE}/30M-p2"):
        model.save_pretrained(d); tok.save_pretrained(d)
        p = f"{d}/tokenizer_config.json"; c = json.load(open(p))
        c["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(c, open(p, "w"))
    json.dump(trainer.state.log_history, open(f"{DRIVE}/loss_log_30M_p2.json", "w"))
    print("saved -> 30M-p2 (local + Drive)", flush=True)

    # ── analysis nhanh: perplexity held-out + owl-rate + distinct ─────────────
    sys.path.insert(0, "/content/tinystory-vn")
    from app.metrics import distinct_n
    from app.perplexity import aggregate_nll, perplexity_from_nll

    model = model.to("cuda").eval()
    tests = [json.loads(l) for l in open("data/tf1/test.jsonl")]
    eos = tok.convert_tokens_to_ids(END)
    gens = []
    for t in tests[:30]:
        cond = t["text"][:t["cond_len"]].rstrip("\n")
        ids = tok(cond + "\n" + SEP, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=320, do_sample=True, temperature=0.8,
                               top_p=0.9, repetition_penalty=1.1,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        txt = tok.decode(o[0], skip_special_tokens=False)
        gens.append(txt.split(SEP, 1)[1].replace(END, "").strip() if SEP in txt else txt)

    lossf = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_seq = []
    for t in tests[:200]:
        enc = encode(t)
        ids = torch.tensor([enc["input_ids"]]).to("cuda")
        lab = torch.tensor([enc["labels"]]).to("cuda")
        with torch.no_grad():
            logits = model(ids).logits
        sl = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        tl = lab[:, 1:].contiguous().view(-1)
        tk = lossf(sl, tl); mask = tl != -100
        if int(mask.sum()):
            per_seq.append((float(tk[mask].sum()), int(mask.sum())))
    ppl = perplexity_from_nll(aggregate_nll(per_seq), sum(x for _, x in per_seq))

    owl_rate = sum(1 for g in gens if "wise old owl" in g.lower()) / len(gens)
    A = {
        "size": "30M-p2", "steps": STEPS, "perplexity": ppl,
        "owl_rate_gen": owl_rate,
        "distinct1": distinct_n(gens, 1), "distinct2": distinct_n(gens, 2),
        "final_loss": trainer.state.log_history[-1].get("train_loss"),
        "gen_stories": [{"cond": tests[i]["text"][:tests[i]["cond_len"]].strip(),
                         "story": gens[i]} for i in range(len(gens))],
    }
    json.dump(A, open(f"{DRIVE}/analysis_30M_p2.json", "w"), ensure_ascii=False)
    print(f"ANALYSIS p2 | ppl={ppl:.2f} | owl-rate gen={owl_rate:.0%} (pha 1 là 90%) | "
          f"distinct2={A['distinct2']:.3f}", flush=True)

    # ── export GGUF p2 (tự dò chkhsh, fail-fast) ───────────────────────────────
    import re
    if not os.path.exists("llama.cpp"):
        sh("git clone -q https://github.com/ggerganov/llama.cpp && pip -q install -r llama.cpp/requirements.txt")
    cmd = f"python llama.cpp/convert_hf_to_gguf.py out/30M-p2 --outfile {DRIVE}/slm-30m-p2.gguf --outtype q8_0"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        m = re.search(r"chkhsh:\s*([0-9a-f]{64})", r.stdout + r.stderr)
        assert m, "convert lỗi, không thấy chkhsh:\n" + r.stderr[-2000:]
        chk = m.group(1)
        bp = "llama.cpp/conversion/base.py"; L = open(bp).read().split("\n")
        if not any(chk in x for x in L):
            for i, ln in enumerate(L):
                if 'raise NotImplementedError("BPE pre-tokenizer was not recognized' in ln:
                    ind = ln[:len(ln) - len(ln.lstrip())]
                    L[i] = f'{ind}if chkhsh == "{chk}": return "gpt-2"\n{ln}'; break
            open(bp, "w").write("\n".join(L))
            print(f"patched chkhsh {chk[:12]}... -> gpt-2", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert r.returncode == 0, "convert vẫn lỗi:\n" + r.stderr[-2000:]
    open(f"{DRIVE}/Modelfile-30M-p2", "w").write(
        'FROM ./slm-30m-p2.gguf\nTEMPLATE """{{ .Prompt }}\\n<|story|>"""\n'
        'PARAMETER temperature 0.8\nPARAMETER top_p 0.9\nPARAMETER repeat_penalty 1.1\n'
        'PARAMETER stop "<|end|>"\nPARAMETER num_ctx 512\n')
    print("DONE. artifacts:", sorted(x for x in os.listdir(DRIVE) if "p2" in x), flush=True)


if __name__ == "__main__":
    main()
