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
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, filepath, special_tokens=None):
        with open(filepath, "rb") as f:
            vocab, merges = pickle.load(f)
        cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens = [regex.escape(special_token) for special_token in special_tokens]
            split_pattern = "|".join(special_tokens)
            splits = regex.split("(" + split_pattern + ")", text)
        else:
            splits = [text]
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ans: list[int] = []
        for chunck in splits:
            if chunck in self.special_tokens:
                ans.append(self.bytes_to_id[chunck.encode()])
                continue
            words = regex.findall(pattern, chunck)
            for word in words:
                bytes_word = self._merge_word(word)
                ans = ans + [self.bytes_to_id[word] for word in bytes_word]
        return ans

    def decode(self, ids: list[int]) -> str:
        bytes_str = b""
        for id in ids:
            bytes_str += self.vocab[id]
        return bytes_str.decode(errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def _merge_word(self, word):
        bytes_word: tuple[bytes, ...] = tuple(word.encode()[i : i + 1] for i in range(len(word.encode())))
        for merge in self.merges:
            outlist = []
            i = 0
            while i < len(bytes_word):
                if i + 1 < len(bytes_word) and (bytes_word[i], bytes_word[i + 1]) == merge:
                    outlist.append(bytes_word[i] + bytes_word[i + 1])
                    i += 2
                else:
                    outlist.append(bytes_word[i])
                    i += 1
            bytes_word = tuple(outlist)
        return bytes_word
