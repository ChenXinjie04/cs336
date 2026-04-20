import os
from typing import BinaryIO
from multiprocessing import Pool
import re
import regex
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass

type StrWordCounter = Counter[str]
type ByteWordCounter = Counter[tuple[bytes, ...]]
type BytePairCounter = Counter[tuple[bytes, bytes]]
type WordEntryList = list[tuple[tuple[bytes, ...], int]]
type PairTable = defaultdict[tuple[bytes, bytes], set[int]]


@dataclass
class BpeState:
    pair_to_word_ids: PairTable
    word_id_to_entry: WordEntryList
    pairs_counter: BytePairCounter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def encode_to_word(str_word_counter: StrWordCounter) -> ByteWordCounter:
    """
    Change str counter into bytes tuple counter. Each str split into bytes tuple.
    Preprocessing for max_pair count and merge.
    Empty string are skipped and do not appear in output.
    {"abc": 3, "aa": 2} -> {(b'a', b'b', b'c'):3, (b'a', b'a'): 2}
    {"": 2} -> {}
    """
    word_table = Counter()
    for key in str_word_counter:
        if key == "":
            continue
        if len(key.encode()) < 2:
            continue
        bytes_key = key.encode()
        bytes_tuple = tuple([bytes_key[i : i + 1] for i in range(len(bytes_key))])
        word_table[bytes_tuple] = str_word_counter[key]
    return word_table


def pre_token(input_path, start, end):
    special_token = "<|endoftext|>"
    special_token = re.escape(special_token)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="strict")
        splits = re.split(special_token, chunk)
        pre_tokens = []
        for document in splits:
            pre_tokens += regex.findall(pattern, document)
        string_counter = Counter(pre_tokens)
        bytes_counter = encode_to_word(string_counter)
    return bytes_counter


def tupecnt2paircnt(tuple_map):
    pairs_counter = Counter()
    for word, cnt in tuple_map.items():
        for pair_tuple in zip(word[:-1], word[1:]):
            pairs_counter[pair_tuple] += cnt
    return pairs_counter


def init_merge_state(byte_word_counter: ByteWordCounter) -> BpeState:
    pair_to_word_ids = defaultdict(set)
    word_id_to_entry = []
    pairs_counter = Counter()
    for idx, (key, value) in enumerate(byte_word_counter.items()):
        word_id_to_entry.append((key, value))
        for i in range(len(key) - 1):
            pair = (key[i], key[i + 1])
            pair_to_word_ids[pair].add(idx)
            pairs_counter[pair] += value
    return BpeState(pair_to_word_ids, word_id_to_entry, pairs_counter)


def _merge_key(state: BpeState, max_pair: tuple[bytes, bytes]):
    pair_to_word_ids = state.pair_to_word_ids
    word_id_to_entry = state.word_id_to_entry
    pairs_counter = state.pairs_counter
    ids = pair_to_word_ids[max_pair].copy()
    for idx in ids:
        (tuple_word, cnt) = word_id_to_entry[idx]
        list_word = []
        old_set = set()
        new_set = set()
        i = 0
        while i < len(tuple_word) - 1:
            old_pair = (tuple_word[i], tuple_word[i + 1])
            old_set.add(old_pair)
            pairs_counter[old_pair] -= cnt
            if pairs_counter[old_pair] == 0:
                del pairs_counter[old_pair]
            if tuple_word[i : i + 2] == max_pair:
                list_word.append(tuple_word[i] + tuple_word[i + 1])
                i += 1
                if i + 1 <= len(tuple_word) - 1:
                    old_pair = (tuple_word[i], tuple_word[i + 1])
                    old_set.add(old_pair)
                    pairs_counter[old_pair] -= cnt
                    if pairs_counter[old_pair] == 0:
                        del pairs_counter[old_pair]
            else:
                list_word.append(tuple_word[i])
            i += 1
        if i == len(tuple_word) - 1:
            list_word.append(tuple_word[-1])
        tuple_word = tuple(list_word)
        for i in range(len(tuple_word) - 1):
            new_pair = (tuple_word[i], tuple_word[i + 1])
            new_set.add(new_pair)
            pairs_counter[new_pair] += cnt
        word_id_to_entry[idx] = (tuple_word, cnt)
        for p in old_set - new_set:
            pair_to_word_ids[p].remove(idx)
            if not pair_to_word_ids[p]:
                del pair_to_word_ids[p]
        for p in new_set - old_set:
            pair_to_word_ids[p].add(idx)


def merge(byte_word_counter: ByteWordCounter, iteration: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Run BPE by iteration rounds; each round finds the most frequent pair and merges it.
    Break tie by lexicographically greater pair(tuple comparison).
    byte_word_counter: will update in place.
    pairs_counter: will update in place.
    merges[i] is the pair merged at round i; vocab[256 + i] is that pair concatencated.
    """
    vocab = {}
    merges = []
    start_id = 256
    state = init_merge_state(byte_word_counter)
    pairs_counter = state.pairs_counter
    for _ in range(iteration):
        if len(pairs_counter) == 0:
            break
        max_pair = max(pairs_counter, key=lambda x: (pairs_counter[x], x))
        merges.append(max_pair)
        new_vocab = max_pair[0] + max_pair[1]
        vocab[start_id] = new_vocab
        start_id += 1
        _merge_key(state, max_pair)
    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    iteration = vocab_size - len(special_tokens) - 256
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    args = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(num_processes) as p:
        ans = p.starmap(pre_token, args)
    tuples_counter = Counter()
    for c in ans:
        tuples_counter += c
    if len(tuples_counter) == 0:
        return {}, []
    vocab, merges = merge(tuples_counter, iteration)
    for i in range(256):
        vocab[i] = bytes([i])
    for next_token_id, token in enumerate(special_tokens, start=len(vocab)):
        vocab[next_token_id] = token.encode()
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        os.path.expanduser("~/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"), 1000, ["<|endoftext|>"]
    )
    with open("tokenization_result.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)
