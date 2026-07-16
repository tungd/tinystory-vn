"""ORPO train cho SLM 30M-p2 trên Colab T4 (TRL), chạy headless hoặc trong notebook.

Đọc preference pairs (RLAIF, từ scripts/gen_preference_pairs.py), train ORPO
khởi từ 30M-p2, guard hồi quy bằng perplexity held-out, lưu 30M-orpo + GGUF.

Chạy trên Colab (sau khi mount Drive):
    python -m trieulh.scripts.orpo_train            # tự dò DRIVE + pairs

Spec: docs/superpowers/specs/2026-07-11-slm-orpo-alignment-design.md
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import deque

SEP, END = "<|story|>", "<|end|>"
SEQ_LEN = 512


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


def heldout_ppl(model, tok, test_path, n=100):
    """Perplexity held-out (loss-mask conditioning) - guard hồi quy sau ORPO."""
    import torch
    lossf = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    total_nll = total_tok = 0.0
    for line in list(open(test_path))[:n]:
        row = json.loads(line)
        ids = tok(row["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
        n_cond = min(len(tok(row["text"][:row["cond_len"]])["input_ids"]), len(ids))
        lab = [-100] * n_cond + ids[n_cond:]
        t_ids = torch.tensor([ids]).to(model.device)
        t_lab = torch.tensor([lab]).to(model.device)
        with torch.no_grad():
            logits = model(t_ids).logits
        sl = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        tl = t_lab[:, 1:].contiguous().view(-1)
        tk = lossf(sl, tl); mask = tl != -100
        if int(mask.sum()):
            total_nll += float(tk[mask].sum()); total_tok += int(mask.sum())
    return math.exp(total_nll / max(1, total_tok))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None, help="pairs.jsonl (mặc định: DRIVE/orpo/pairs.jsonl)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="hệ số odds-ratio (lambda của ORPO)")
    args = ap.parse_args(argv)

    subprocess.run("pip -q install trl datasets", shell=True)
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from trl import ORPOConfig, ORPOTrainer

    DRIVE = find_drive_root()
    assert DRIVE, "không tìm thấy thư mục Drive chứa ckpt_30M"
    pairs_path = args.pairs or f"{DRIVE}/orpo/pairs.jsonl"
    src = f"{DRIVE}/30M-p2"
    assert os.path.isdir(src), f"thiếu {src} - chạy pha 2 trước"
    assert os.path.exists(pairs_path), f"thiếu {pairs_path} - chạy gen_preference_pairs trước"
    print("DRIVE =", DRIVE, "| pairs =", pairs_path, flush=True)

    # pairs -> format ORPO, khớp đúng template train của SLM
    rows = [json.loads(l) for l in open(pairs_path)]
    rows = [r for r in rows if not r.get("filtered")]
    data = Dataset.from_list([{
        "prompt": r["prompt"] + "\n" + SEP,
        "chosen": r["chosen"].strip() + END,
        "rejected": r["rejected"].strip() + END,
    } for r in rows])
    print(f"pairs dùng được: {len(data)}", flush=True)
    assert len(data) >= 300, "quá ít pairs - kiểm tra lại gen_preference_pairs"

    tok = PreTrainedTokenizerFast(tokenizer_file=f"{src}/tokenizer.json",
            unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")
    model = AutoModelForCausalLM.from_pretrained(src).to("cuda")

    test_path = f"{DRIVE}/data_tf1_v2/test.jsonl"
    ppl_before = heldout_ppl(model.eval(), tok, test_path)
    print(f"ppl held-out TRƯỚC ORPO: {ppl_before:.2f}", flush=True)

    cfg = ORPOConfig(
        output_dir=f"{DRIVE}/ckpt_30M_orpo", num_train_epochs=args.epochs,
        learning_rate=args.lr, beta=args.beta,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        max_length=SEQ_LEN, max_prompt_length=192,
        logging_steps=10, save_strategy="epoch", fp16=True, report_to=[],
    )
    trainer = ORPOTrainer(model=model, args=cfg, train_dataset=data,
                          processing_class=tok)
    trainer.train()

    ppl_after = heldout_ppl(model.eval(), tok, test_path)
    drift = (ppl_after - ppl_before) / ppl_before
    print(f"ppl held-out SAU ORPO: {ppl_after:.2f} (drift {drift:+.1%}, guard <= +10%)", flush=True)
    if drift > 0.10:
        print("CẢNH BÁO: perplexity tăng quá 10% - cân nhắc giảm epoch/LR rồi chạy lại", flush=True)

    for d in ("out/30M-orpo", f"{DRIVE}/30M-orpo"):
        model.save_pretrained(d); tok.save_pretrained(d)
        p = f"{d}/tokenizer_config.json"; c = json.load(open(p))
        c["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(c, open(p, "w"))
    json.dump(trainer.state.log_history, open(f"{DRIVE}/loss_log_30M_orpo.json", "w"))
    json.dump({"ppl_before": ppl_before, "ppl_after": ppl_after, "n_pairs": len(data)},
              open(f"{DRIVE}/orpo_summary.json", "w"))
    print("saved -> 30M-orpo", flush=True)

    # export GGUF (tự dò chkhsh, fail-fast - như export pha 2)
    if not os.path.exists("llama.cpp"):
        subprocess.run("git clone -q https://github.com/ggerganov/llama.cpp && pip -q install -r llama.cpp/requirements.txt", shell=True)
    cmd = f"python llama.cpp/convert_hf_to_gguf.py out/30M-orpo --outfile {DRIVE}/slm-30m-orpo.gguf --outtype q8_0"
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
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert r.returncode == 0, "convert vẫn lỗi:\n" + r.stderr[-2000:]
    open(f"{DRIVE}/Modelfile-30M-orpo", "w").write(
        'FROM ./slm-30m-orpo.gguf\nTEMPLATE """{{ .Prompt }}\\n<|story|>"""\n'
        'PARAMETER temperature 0.8\nPARAMETER top_p 0.9\nPARAMETER repeat_penalty 1.1\n'
        'PARAMETER stop "<|end|>"\nPARAMETER num_ctx 512\n')
    print("DONE:", sorted(x for x in os.listdir(DRIVE) if "orpo" in x.lower()), flush=True)


if __name__ == "__main__":
    main()
