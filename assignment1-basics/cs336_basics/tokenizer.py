import pickle
import regex
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        special_tokens = [] if special_tokens is None else special_tokens
        index = max(vocab) + 1
        for special_token in special_tokens:
            if special_token not in vocab.values():
                vocab[index] = special_token
                index += 1
        self.vocab = vocab
        self.merges = merges
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = set(special_tokens)
        self.pair_dict = {merge: idx for idx, merge in enumerate(merges)}
        self.end_of_token = len(vocab)
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_tokens = [regex.escape(special_token) for special_token in special_tokens]
        self.split_pattern = "(" + "|".join(special_tokens) + ")"
        self._merge_cache = {}
        self.pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, filepath, special_tokens=None):
        with open(filepath, "rb") as f:
            vocab, merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            splits = regex.split(self.split_pattern, text)
        else:
            splits = [text]
        ans: list[int] = []
        for chunck in splits:
            if chunck in self.special_tokens:
                ans.append(self.bytes_to_id[chunck.encode()])
                continue
            words = self.pattern.findall(chunck)
            for word in words:
                if word in self._merge_cache:
                    ids = self._merge_cache[word]
                else:
                    ids = self._merge_word(word)
                ans.extend(ids)
        return ans

    def decode(self, ids: list[int]) -> str:
        bytes_str = b""
        for id in ids:
            new_str = self.vocab[id]
            bytes_str += new_str
        return bytes_str.decode(errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def _merge_word(self, word):
        bytes_word = word.encode()
        bytes_list = [bytes_word[i : i + 1] for i in range(len(bytes_word))]
        end_of_token = self.end_of_token
        pair_dict = self.pair_dict
        while True:
            new_bytes_list = []
            merge_pair = ()
            min_id = end_of_token
            for i in range(len(bytes_list) - 1):
                cur_pair = (bytes_list[i], bytes_list[i + 1])
                if cur_pair not in pair_dict:
                    continue
                if pair_dict[cur_pair] < min_id:
                    min_id = pair_dict[cur_pair]
                    merge_pair = cur_pair
            if min_id == end_of_token:
                break
            i = 0
            while i < len(bytes_list):
                if i + 1 < len(bytes_list) and (bytes_list[i], bytes_list[i + 1]) == merge_pair:
                    new_bytes_list.append(bytes_list[i] + bytes_list[i + 1])
                    i += 2
                else:
                    new_bytes_list.append(bytes_list[i])
                    i += 1
            bytes_list = new_bytes_list
        ids = [self.bytes_to_id[bytes_word] for bytes_word in bytes_list]
        self._merge_cache[word] = ids
        return ids
