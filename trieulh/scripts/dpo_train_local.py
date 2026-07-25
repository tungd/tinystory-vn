"""DPO trial LOCAL trên Apple Silicon (MPS) - train thử với pairs sẵn có.

Preference optimization bằng DPO (Rafailov et al. 2023). Lưu ý: chọn DPO thay ORPO
vì trl 1.8 (bản rewrite) đã bỏ ORPOTrainer; DPO là biến thể kinh điển tương đương,
cùng định dạng data (chosen/rejected). Khác biệt: DPO cần reference model (bản đóng
băng của chính model) - với 36M param là rẻ.

Khởi từ out/30M-p2 (HF), train trên data/orpo/pairs.jsonl, guard perplexity held-out,
lưu out/30M-dpo. Không export GGUF (để bản Colab lo, hoặc convert sau).

Usage:
    python -m trieulh.scripts.dpo_train_local --epochs 2 --lr 5e-6
"""
import argparse
import json
import math

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from trl import DPOConfig, DPOTrainer

SEP, END, SEQ_LEN = "<|story|>", "<|end|>", 512


def heldout_ppl(model, tok, test_path, device, n=100):
    lossf = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    tot_nll = tot_tok = 0.0
    model.eval()
    for line in list(open(test_path))[:n]:
        row = json.loads(line)
        ids = tok(row["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
        n_cond = min(len(tok(row["text"][:row["cond_len"]])["input_ids"]), len(ids))
        lab = [-100] * n_cond + ids[n_cond:]
        t_ids = torch.tensor([ids]).to(device)
        t_lab = torch.tensor([lab]).to(device)
        with torch.no_grad():
            logits = model(t_ids).logits
        sl = logits[:, :-1].reshape(-1, logits.size(-1))
        tl = t_lab[:, 1:].reshape(-1)
        tk = lossf(sl, tl); mask = tl != -100
        if int(mask.sum()):
            tot_nll += float(tk[mask].sum()); tot_tok += int(mask.sum())
    return math.exp(tot_nll / max(1, tot_tok))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/30M-p2")
    ap.add_argument("--pairs", default="data/orpo/pairs.jsonl")
    ap.add_argument("--test", default="data/tf1/test.jsonl")
    ap.add_argument("--out", default="out/30M-dpo")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--min-pairs", type=int, default=50)
    args = ap.parse_args(argv)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows if not r.get("filtered")]
    print(f"pairs giữ được: {len(rows)}", flush=True)
    assert len(rows) >= args.min_pairs, f"quá ít pairs ({len(rows)} < {args.min_pairs})"
    data = Dataset.from_list([{
        "prompt": r["prompt"] + "\n" + SEP,
        "chosen": r["chosen"].strip() + END,
        "rejected": r["rejected"].strip() + END,
    } for r in rows])

    tok = PreTrainedTokenizerFast(tokenizer_file=f"{args.model}/tokenizer.json",
            unk_token="<|unk|>", pad_token="<|pad|>", eos_token="<|end|>")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.model).to(device)  # reference đóng băng

    ppl_before = heldout_ppl(model, tok, args.test, device)
    print(f"ppl held-out TRƯỚC DPO: {ppl_before:.3f}", flush=True)

    cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        beta=args.beta, per_device_train_batch_size=4, gradient_accumulation_steps=2,
        max_length=SEQ_LEN,
        logging_steps=5, save_strategy="no", report_to=[], bf16=False, fp16=False,
    )
    trainer = DPOTrainer(model=model, ref_model=ref, args=cfg,
                         train_dataset=data, processing_class=tok)
    trainer.train()

    ppl_after = heldout_ppl(model, tok, args.test, device)
    drift = (ppl_after - ppl_before) / ppl_before
    print(f"ppl held-out SAU DPO: {ppl_after:.3f} (drift {drift:+.1%}; guard tham khảo <= +10%)", flush=True)

    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    p = f"{args.out}/tokenizer_config.json"; c = json.load(open(p))
    c["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(c, open(p, "w"))
    json.dump({"method": "DPO (trl 1.8; ORPO unavailable)", "ppl_before": ppl_before,
               "ppl_after": ppl_after, "n_pairs": len(rows), "epochs": args.epochs, "lr": args.lr},
              open(f"{args.out}/dpo_trial_summary.json", "w"))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
