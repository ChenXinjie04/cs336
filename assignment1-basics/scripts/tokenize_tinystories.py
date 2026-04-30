from cs336_basics.tokenizer import Tokenizer
import pickle

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files("./tokenizers/tokenizer_tiny_stories.pkl")
    with open("./data/tinystories_sample_1M.txt", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    print(ids)
