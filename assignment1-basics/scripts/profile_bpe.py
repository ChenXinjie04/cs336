from cs336_basics.tokenizer import train_bpe
import os
import cProfile


def profile_bpe():
    train_bpe(
        os.path.expanduser("~/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"), 1000, ["<|endoftext|>"]
    )


if __name__ == "__main__":
    cProfile.run("profile_bpe()", "tokenizer.prof")
