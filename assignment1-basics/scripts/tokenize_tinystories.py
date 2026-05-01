from cs336_basics.tokenizer import Tokenizer
import numpy as np

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files("./tokenizers/tokenizer_tiny_stories.pkl")
    with open("./data/tinystories_sample.txt", encoding="utf-8") as f:
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save("tokens.npy", arr)
