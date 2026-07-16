"""Train a small BPE tokenizer on the fable corpus."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from trieulh.scripts.tf1_pretrain.format import SEP, END, PAD


def train_bpe(texts, vocab_size: int, special_tokens):
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|unk|>", PAD, *special_tokens],
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tf1/train.jsonl")
    ap.add_argument("--out", default="data/tf1/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=12000)
    args = ap.parse_args(argv)

    def it():
        with open(args.data) as f:
            for line in f:
                yield json.loads(line)["text"]

    tok = train_bpe(it(), args.vocab_size, [SEP, END])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer ({args.vocab_size} vocab) to {args.out}")


if __name__ == "__main__":
    main()
