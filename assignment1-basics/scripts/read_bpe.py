import pickle

if __name__ == "__main__":
    with open("./tokenizer_tiny_stories.pkl", "rb") as f:
        (vocab, merges) = pickle.load(f)
    # 前 20 个（通常是 byte-level 基础 token）
    print(f"{len(vocab)=}")
    for i in range(20):
        print(i, vocab[i])

    # 后 20 个（通常是训练后期 merge 出的长 token，最能反映语料特点）
    for i in range(len(vocab) - 20, len(vocab)):
        print(i, vocab[i])

    print(merges[:10])  # 最早的 merge，通常是高频字符对，比如 (b' ', b't')
    print(merges[-10:])
    by_len = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)
    by_len = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)
    for tid, tok in by_len[:30]:
        print(tid, len(tok), tok)
