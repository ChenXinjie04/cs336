from cs336_basics.tokenizer import Tokenizer
import time

if __name__ == "__main__":
    start = time.perf_counter()
    tokenizer = Tokenizer.from_files("./tokenizers/tokenizer_tiny_stories.pkl")
    with open("./data/tinystories_sample_5M.txt", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    print("total time", time.perf_counter() - start)
