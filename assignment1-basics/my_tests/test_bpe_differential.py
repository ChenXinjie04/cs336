from cs336_basics import train_bpe
from cs336_basics import train_bpe_fast
from pathlib import Path

special_tokens = ["<|endoftext|>"]
SMOKE_PATH = Path(__file__).parent.parent / "data" / "small.txt"
assert SMOKE_PATH.exists(), f"smoke data not found: {SMOKE_PATH}"


def test_smoke():
    vocab_size = 10000
    (slow_vocab, slow_merges) = train_bpe.train_bpe(SMOKE_PATH, vocab_size, special_tokens)
    (fast_vocab, fast_merges) = train_bpe_fast.train_bpe(SMOKE_PATH, vocab_size, special_tokens)
    vocab_idx = -1
    merge_idx = -1
    assert len(slow_vocab) == len(fast_vocab), "different length"
    assert len(slow_merges) == len(fast_merges)
    for i in range(len(slow_vocab)):
        if slow_vocab[i] != fast_vocab[i]:
            vocab_idx = i
            break
    for i, _ in enumerate(slow_merges):
        if slow_merges[i] != fast_merges[i]:
            merge_idx = i
            break
    assert vocab_idx == -1, f"{slow_vocab[vocab_idx] = }, {fast_vocab[vocab_idx] = }"
    assert merge_idx == -1, f"{slow_merges[merge_idx] = }, {fast_merges[merge_idx] = }"
