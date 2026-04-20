import os
from cs336_basics import tokenizer
from cs336_basics import tokenizer_fast
from pathlib import Path

special_tokens = ["<|endoftext|>"]
SMOKE_PATH = Path(__file__).parent.parent / "data" / "smoke.txt"
assert SMOKE_PATH.exists(), f"smoke data not found: {SMOKE_PATH}"


def test_smoke():
    vocab_size = 1022
    (slow_vocab, slow_merges) = tokenizer.train_bpe(SMOKE_PATH, vocab_size, special_tokens)
    (fast_vocab, fast_merges) = tokenizer_fast.train_bpe(SMOKE_PATH, vocab_size, special_tokens)
    vocab_idx = -1
    merge_idx = -1
    assert len(slow_vocab) == len(fast_vocab), "different length"
    for i in range(vocab_size):
        if slow_vocab[i] != fast_vocab[i]:
            vocab_idx = i
            break
    for i in range(vocab_size - 256 - len(special_tokens)):
        if slow_merges[i] != fast_merges[i]:
            merge_idx = i
            break
    assert vocab_idx == -1, f"{slow_vocab[vocab_idx] = }, {fast_vocab[vocab_idx] = }"
    assert merge_idx == -1, f"{slow_merges[merge_idx] = }, {fast_merges[merge_idx] = }"
