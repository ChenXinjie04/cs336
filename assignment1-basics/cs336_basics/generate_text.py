from cs336_basics.tokenizer import Tokenizer
import pickle
from cs336_basics.train_script import decode


with open("./tokenizers/tokenizer_tiny_stories.pkl", "rb") as f:
    vocab, merges = pickle.load(f)
last_id = max(vocab)
t = Tokenizer(vocab, merges, ["<|endoftext|>"])
s = input("user_input: ")
user_input = t.encode(s)
lm_output = decode(user_input, 1.0, 180, last_id, 0.9)
assert 9999 not in lm_output
print("model_output: " + t.decode(lm_output))
