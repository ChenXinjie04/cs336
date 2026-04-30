from cs336_basics.tokenizer import Tokenizer


def test_encode():
    vocab = {1: b"hello", 2: b"app", 3: b"le", 4: b"."}
    merges = [(b"h", b"e"), (b"l", b"l"), (b"he", b"ll"), (b"hell", b"o"), (b"a", b"p"), (b"ap", b"p"), (b"l", b"e")]
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.encode("hello.apple") == [
        1,
        4,
        2,
        3,
    ]

def test_unicode_roundtrip():
