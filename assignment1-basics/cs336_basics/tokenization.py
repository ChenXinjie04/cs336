import os
from typing import BinaryIO
from multiprocessing import Pool
import re
import regex
from collections import Counter

FILE_PATH = "../data/TinyStoriesV2-GPT4-valid.txt"
type StrWordCounter = Counter[str]
type ByteWordCounter = Counter[tuple[bytes, ...]]
type BytePairCounter = Counter[tuple[bytes, bytes]]


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
        bytes_key = key.encode()
        bytes_tuple = tuple([bytes_key[i : i + 1] for i in range(len(bytes_key))])
        word_table[bytes_tuple] = str_word_counter[key]
    return word_table


def pre_token(start, end):
    special_token = "<|endoftext|>"
    special_token = re.escape(special_token)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(FILE_PATH, "rb") as f:
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


def _merge_key(tuples_counter, pairs_counter):
    return


def merge(
    byte_word_counter: ByteWordCounter, pairs_counter: BytePairCounter, iteration: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    for _ in range(iteration):
        max_pair = max(pairs_counter, key=lambda x: (pairs_counter[x], x))
        merges.append(max_pair)
        vocab[start_id] = max_pair[0] + max_pair[1]
        start_id += 1
    raise NotImplementedError


if __name__ == "__main__":
    print("hello")
    ## Usage
    iteration = 4
    with open(FILE_PATH, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    args = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(num_processes) as p:
        ans = p.starmap(pre_token, args)
    tuples_counter = Counter()
    for c in ans:
        tuples_counter += c
    pairs_counter = tupecnt2paircnt(tuples_counter)
    vocab, merges = merge(tuples_counter, pairs_counter, iteration)
