import pytest

tokenizers = pytest.importorskip("tokenizers")
from trieulh.scripts.train_tokenizer import train_bpe
from trieulh.scripts.tf1_pretrain.format import SEP, END, PAD


def test_train_bpe_roundtrip_and_specials():
    texts = ["a fox shared food. moral: be kind."] * 50
    tok = train_bpe(texts, vocab_size=300, special_tokens=[SEP, END, PAD])
    assert tok.token_to_id(SEP) is not None
    ids = tok.encode(f"a fox {SEP} be kind {END}").ids
    assert len(ids) > 0
