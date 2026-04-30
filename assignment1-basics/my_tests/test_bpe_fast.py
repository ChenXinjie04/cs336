from cs336_basics.train_bpe_fast import BpeState, encode_to_word, init_merge_state, merge, tupecnt2paircnt, _merge_key
from collections import Counter, defaultdict


def test_encode_to_word():
    input = Counter({"abc": 2, "ab": 3, " ": 5})
    output = Counter({(b"a", b"b", b"c"): 2, (b"a", b"b"): 3})
    assert encode_to_word(input) == output


def test_encode_to_word_empty():
    input = Counter({"": 2, "ab": 3})
    output = Counter({(b"a", b"b"): 3})
    assert encode_to_word(input) == output


def test_tupecnt2paircnt():
    input = {(b"a", b"c", b"d"): 2}
    output = Counter({(b"a", b"c"): 2, (b"c", b"d"): 2})
    assert tupecnt2paircnt(input) == output


def test_merge_basic():
    byte_word_counter = Counter(
        {
            (b"l", b"o", b"w"): 5,
            (b"l", b"o", b"w", b"e", b"r"): 2,
            (b"w", b"i", b"d", b"e", b"s", b"t"): 3,
            (b"n", b"e", b"w", b"e", b"s", b"t"): 6,
        }
    )
    pairs_counter = Counter(
        {
            (b"l", b"o"): 7,
            (b"o", b"w"): 7,
            (b"w", b"e"): 8,
            (b"e", b"r"): 2,
            (b"w", b"i"): 3,
            (b"i", b"d"): 3,
            (b"d", b"e"): 3,
            (b"e", b"s"): 9,
            (b"s", b"t"): 9,
            (b"n", b"e"): 6,
            (b"e", b"w"): 6,
        }
    )
    output = (
        {256: b"st", 257: b"est", 258: b"ow", 259: b"low", 260: b"west", 261: b"ne"},
        [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")],
    )
    ans = merge(byte_word_counter, 6)
    assert ans == output


def test_merge_corner_case():
    byte_word_counter = Counter(
        {
            (b"a", b"a", b"a"): 3,
            (b"a", b"b", b"a", b"b"): 2,
            (b"a", b"b", b"x", b"x", b"a", b"b", b"x", b"x", b"a", b"b"): 4,
        }
    )
    pairs_counter = Counter(
        {
            (b"a", b"a"): 6,
            (b"a", b"b"): 16,
            (b"b", b"a"): 2,
            (b"b", b"x"): 8,
            (b"x", b"x"): 8,
            (b"x", b"a"): 8,
        }
    )
    output = (
        {256: b"ab", 257: b"xx", 258: b"xxab", 259: b"aa", 260: b"xxabxxab", 261: b"abxxabxxab", 262: b"aaa"},
        [
            (b"a", b"b"),
            (b"x", b"x"),
            (b"xx", b"ab"),
            (b"a", b"a"),
            (b"xxab", b"xxab"),
            (b"ab", b"xxabxxab"),
            (b"aa", b"a"),
        ],
    )
    ans = merge(byte_word_counter, 7)
    assert ans == output


def test_merge_out_of_pairs():
    byte_word_counter = Counter(
        {
            (b"a", b"a", b"a", b"a"): 3,
        }
    )
    pairs_counter = Counter({(b"a", b"a"): 9})
    output = (
        {256: b"aa", 257: b"aaaa"},
        [(b"a", b"a"), (b"aa", b"aa")],
    )
    ans = merge(byte_word_counter, 4)
    assert ans == output


def test_init_merge_state_basic():
    input = Counter({(b"a", b"b", b"a", b"b"): 3, (b"x", b"y"): 4})
    state = init_merge_state(input)
    assert state.pair_to_word_ids[(b"a", b"b")] == {0}
    assert state.pair_to_word_ids[(b"x", b"y")] == {1}
    assert state.pair_to_word_ids[(b"b", b"a")] == {0}
    assert state.word_id_to_entry[0] == ((b"a", b"b", b"a", b"b"), 3)
    assert state.word_id_to_entry[1] == ((b"x", b"y"), 4)
    assert state.pairs_counter[(b"a", b"b")] == 6
    assert state.pairs_counter[(b"b", b"a")] == 3
    assert state.pairs_counter[(b"x", b"y")] == 4


def test_merge_key_basic():
    input_state = BpeState(
        defaultdict(set, {(b"a", b"b"): {0}, (b"b", b"a"): {0}}),
        [((b"a", b"b", b"a", b"b"), 3)],
        Counter({(b"a", b"b"): 6, (b"b", b"a"): 3}),
    )
    max_pair = (b"a", b"b")
    output_state = BpeState(
        defaultdict(set, {(b"ab", b"ab"): {0}}),
        [((b"ab", b"ab"), 3)],
        Counter({(b"ab", b"ab"): 3}),
    )
    _merge_key(input_state, max_pair)
    assert input_state == output_state
