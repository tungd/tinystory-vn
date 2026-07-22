"""Pretrain SLM 60M from scratch trên Colab T4, điều khiển qua Colab CLI (chế độ A).

Scale-up từ 30M theo kết luận campaign (nâng floor = pretrain): hidden 512->768,
heads 8->12 (kv 4), seq 512->1024 (mở khóa length control), data 400k->FULL TF1
(~2.5-2.8M truyện sau lọc ~ 1 epoch Chinchilla-cho-60M), giữ nguyên recipe đã
validate: tokenizer 12k, WSD, AdamW(0.9,0.95) wd 0.1, clip 1.0, can thiệp data v2
(cap "wise old owl" 10% + slot-dropout teaching/outcome 0.15).

Resume-first: checkpoint mỗi 500 step vào {DRIVE}/ckpt_60M (Drive mount chế độ A),
tự resume từ checkpoint mới nhất; corpus pack thành numpy int16 cache trên Drive
(build một lần ~2.2GB, các phiên sau copy về VM trong vài phút).

Chạy: python colab_pretrain_60m.py [--steps 15000] (mọi print đều flush cho CLI).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import deque

# ── config 60M ────────────────────────────────────────────────────────────────
HIDDEN, LAYERS, HEADS, KV_HEADS, FFN = 768, 8, 12, 4, 2048
VOCAB, SEQ_LEN = 12000, 1024
TRAIN_N = 3_000_000            # full TF1 (thực tế ~2.5-2.8M sau lọc + dedup)
BATCH_SIZE, GRAD_ACCUM = 16, 8  # eff 128 seq/step
PEAK_LR, ADAM_BETAS, WEIGHT_DECAY, GRAD_CLIP = 3e-3, (0.9, 0.95), 0.1, 1.0
WARMUP_FRAC, DECAY_FRAC = 0.02, 0.15
LOG_EVERY, SAVE_EVERY = 25, 500
SEP, END = "<|story|>", "<|end|>"


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(f"FAILED: {cmd}")


def find_drive_root():
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


def build_or_restore_corpus(DRIVE, tok):
    """Corpus 60M dạng packed numpy: tokens (int16 flat), offsets, cond_tok_lens."""
    import numpy as np
    cache = f"{DRIVE}/data_tf1_60m"
    local = "/content/data60"
    os.makedirs(local, exist_ok=True)
    names = ["tokens.npy", "offsets.npy", "condlens.npy"]
    if all(os.path.exists(f"{cache}/{n}") for n in names):
        print("khôi phục corpus 60M từ cache Drive...", flush=True)
        for n in names + ["test.jsonl"]:
            if not os.path.exists(f"{local}/{n}"):
                shutil.copy(f"{cache}/{n}", f"{local}/{n}")
    else:
        print("build corpus 60M (full TF1, một lần duy nhất)...", flush=True)
        sh(f"python -m trieulh.scripts.prepare_tf1_pretrain --train-n {TRAIN_N} --test-n 500 "
           f"--min-words 60 --max-words 320 "
           f'--cap-phrase "wise old owl" --cap-frac 0.10 '
           f"--slot-dropout teaching=0.15 outcome=0.15 --out data/tf1_60m")
        # pack: encode theo batch -> flat int16 + offsets + cond token lens
        toks_chunks, offsets, condlens = [], [0], []
        batch_txt, batch_cond = [], []
        total = 0

        def flush_batch():
            nonlocal total
            if not batch_txt:
                return
            enc = tok(batch_txt, truncation=True, max_length=SEQ_LEN)["input_ids"]
            enc_c = tok(batch_cond, truncation=True, max_length=SEQ_LEN)["input_ids"]
            for ids, cids in zip(enc, enc_c):
                toks_chunks.append(np.asarray(ids, dtype=np.int16))
                offsets.append(offsets[-1] + len(ids))
                condlens.append(min(len(cids), len(ids)))
                total += 1
            batch_txt.clear(); batch_cond.clear()

        for line in open("data/tf1_60m/train.jsonl"):
            r = json.loads(line)
            batch_txt.append(r["text"])
            batch_cond.append(r["text"][:r["cond_len"]])
            if len(batch_txt) >= 20_000:
                flush_batch()
                if total % 200_000 < 20_000:
                    print(f"  encoded {total}...", flush=True)
        flush_batch()
        tokens = np.concatenate(toks_chunks)
        np.save(f"{local}/tokens.npy", tokens)
        np.save(f"{local}/offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(f"{local}/condlens.npy", np.asarray(condlens, dtype=np.int32))
        shutil.copy("data/tf1_60m/test.jsonl", f"{local}/test.jsonl")
        os.makedirs(cache, exist_ok=True)
        for n in names + ["test.jsonl"]:
            shutil.copy(f"{local}/{n}", f"{cache}/{n}")
        print(f"corpus packed: {total} truyện, {len(tokens)/1e6:.0f}M token -> cache Drive", flush=True)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    a = ap.parse_args()
    STEPS = a.steps

    import numpy as np
    import torch
    assert torch.cuda.is_available(), "session khong co GPU"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    DRIVE = find_drive_root()
    assert DRIVE, "khong tim thay Drive (mount chua?)"
    print("DRIVE =", DRIVE, flush=True)

    if not os.path.exists("/content/tinystory-vn"):
        sh("git clone -q https://github.com/tungd/tinystory-vn.git /content/tinystory-vn")
    os.chdir("/content/tinystory-vn")
    sh("git fetch -q origin && git checkout -q feat/trieulh-60m && git pull -q")
    sh('pip -q install "datasets>=2.20" "tokenizers>=0.19" "transformers>=4.44" accelerate')

    from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                              Trainer, TrainingArguments)
    tok_path = f"{DRIVE}/data_tf1_v2/tokenizer.json"
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_path,
                                  unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")

    local = build_or_restore_corpus(DRIVE, tok)
    tokens = np.load(f"{local}/tokens.npy", mmap_mode="r")
    offsets = np.load(f"{local}/offsets.npy")
    condlens = np.load(f"{local}/condlens.npy")
    n_rows = len(condlens)
    print(f"corpus: {n_rows} truyện, {len(tokens)/1e6:.0f}M token "
          f"(~{len(tokens)/ (BATCH_SIZE*GRAD_ACCUM)/1e3:.0f}k token/step-eff)", flush=True)

    class PackedDS(torch.utils.data.Dataset):
        def __len__(self):
            return n_rows
        def __getitem__(self, i):
            ids = tokens[offsets[i]:offsets[i + 1]].astype(np.int64).tolist()
            nc = int(condlens[i])
            return {"input_ids": ids, "labels": [-100] * nc + ids[nc:]}

    def collator(features):
        pad = tok.pad_token_id
        m = max(len(f["input_ids"]) for f in features)
        fill = lambda s, v: s + [v] * (m - len(s))
        return {
            "input_ids": torch.tensor([fill(f["input_ids"], pad) for f in features]),
            "labels": torch.tensor([fill(f["labels"], -100) for f in features]),
            "attention_mask": torch.tensor(
                [[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in features]),
        }

    # ── model 60M: fresh hoặc resume ──────────────────────────────────────────
    ckpt_dir = f"{DRIVE}/ckpt_60M"
    ckpts = sorted(glob.glob(f"{ckpt_dir}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume_from = ckpts[-1] if ckpts else None
    if resume_from:
        print("resume từ:", resume_from, flush=True)
        model = LlamaForCausalLM.from_pretrained(resume_from)
    else:
        cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                          num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
                          intermediate_size=FFN, max_position_embeddings=SEQ_LEN,
                          tie_word_embeddings=True,
                          bos_token_id=None, eos_token_id=tok.convert_tokens_to_ids(END),
                          pad_token_id=tok.pad_token_id)
        model = LlamaForCausalLM(cfg)
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
        output_dir=ckpt_dir, max_steps=STEPS, fp16=True,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=GRAD_CLIP, logging_steps=LOG_EVERY,
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=2,
        lr_scheduler_type="constant", report_to=[],
        dataloader_num_workers=2, disable_tqdm=True,
    )
    from transformers import TrainerCallback

    class DriveLog(TrainerCallback):
        """Ghi log train lên Drive để theo dõi từ ngoài (poll qua Drive API)."""
        def __init__(self, path):
            self.path = path
        def on_log(self, args_, state, control, logs=None, **kw):
            if logs:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    trainer = Trainer(model=model, args=args, train_dataset=PackedDS(),
                      data_collator=collator, optimizers=(optimizer, scheduler))
    trainer.add_callback(DriveLog(f"{DRIVE}/train60_progress.jsonl"))
    trainer.train(resume_from_checkpoint=resume_from)

    # ── save final + analysis nhanh ───────────────────────────────────────────
    for d in ("out/60M", f"{DRIVE}/60M"):
        model.save_pretrained(d); tok.save_pretrained(d)
        p = f"{d}/tokenizer_config.json"; c = json.load(open(p))
        c["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(c, open(p, "w"))
    json.dump(trainer.state.log_history, open(f"{DRIVE}/loss_log_60M.json", "w"))
    print("saved -> 60M (local + Drive)", flush=True)

    model = model.to("cuda").eval()
    tests = [json.loads(l) for l in open(f"{local}/test.jsonl")]
    eos = tok.convert_tokens_to_ids(END)
    gens = []
    for t in tests[:30]:
        cond = t["text"][:t["cond_len"]].rstrip("\n")
        ids = tok(cond + "\n" + SEP, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=700, do_sample=True, temperature=0.8,
                               top_p=0.9, repetition_penalty=1.1,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        txt = tok.decode(o[0], skip_special_tokens=False)
        gens.append(txt.split(SEP, 1)[1].replace(END, "").strip() if SEP in txt else txt)

    lossf = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_seq = []
    for t in tests[:200]:
        ids_l = tok(t["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
        nc = min(len(tok(t["text"][:t["cond_len"]])["input_ids"]), len(ids_l))
        ids = torch.tensor([ids_l]).to("cuda")
        lab = torch.tensor([[-100] * nc + ids_l[nc:]]).to("cuda")
        with torch.no_grad():
            logits = model(ids).logits
        sl = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        tl = lab[:, 1:].contiguous().view(-1)
        tk = lossf(sl, tl); mask = tl != -100
        if int(mask.sum()):
            per_seq.append((float(tk[mask].sum()), int(mask.sum())))
    import math
    ppl = math.exp(sum(x for x, _ in per_seq) / sum(x for _, x in per_seq))
    owl_rate = sum(1 for g in gens if "wise old owl" in g.lower()) / len(gens)
    A = {"size": "60M", "steps": STEPS, "perplexity": ppl, "owl_rate_gen": owl_rate,
         "final_loss": trainer.state.log_history[-1].get("train_loss"),
         "gen_stories": [{"cond": tests[i]["text"][:tests[i]["cond_len"]].strip(),
                          "story": gens[i]} for i in range(len(gens))]}
    json.dump(A, open(f"{DRIVE}/analysis_60M.json", "w"), ensure_ascii=False)
    print(f"ANALYSIS 60M | ppl={ppl:.2f} | owl={owl_rate:.0%}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
