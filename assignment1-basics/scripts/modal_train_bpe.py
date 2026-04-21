import modal

app = modal.App("example")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .pip_install("regex")
    .pip_install("py-spy")
    .pip_install("gunzip")
    .add_local_python_source("cs336_basics")
)
vol = modal.Volume.from_name("cs336-data", create_if_missing=True)


@app.function(image=image, cpu=4.0, memory=16 * 1024)
def probe():
    import os

    print(os.listdir("/sys/fs/cgroup/memory/"))
    with open("/proc/self/cgroup") as f:
        print(f.read())


@app.function(image=image, volumes={"/data": vol})
def download_tiny_stories_data():
    import subprocess

    subprocess.run(
        "wget -O /data/TinyStoriesV2-GPT4-train.txt --progress=dot:giga https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        shell=True,
        check=True,
    )
    vol.commit()


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def download_owt_data():
    import subprocess

    subprocess.run(
        "gunzip /data/owt_train.txt.gz",
        shell=True,
        check=True,
    )
    vol.commit()


@app.function(image=image, volumes={"/data": vol})
def verify():
    import pickle

    with open("/data/tokenizer.pkl", "rb") as f:
        vocab, merges = pickle.load(f)
        assert len(vocab) == 10000
        for i in range(256):
            assert vocab[i] == bytes([i])
        assert "<|endoftext|>".encode() in vocab.values()
        print(sorted(vocab.values(), key=len, reverse=True)[:20])
        print(sorted(vocab.items())[-20:])
        assert len(merges) == 10000 - 256 - 1


@app.function(image=image, volumes={"/data": vol}, cpu=4.0, memory=30 * 1024, timeout=1800)
def train_tiny_stories():
    from cs336_basics.train_bpe_fast import train_bpe
    import time
    import pickle

    start = time.perf_counter()
    ans = train_bpe("/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    print(time.perf_counter() - start)
    with open("/data/tokenizer_tiny_stories.pkl", "wb") as f:
        pickle.dump(ans, f)
    vol.commit()


@app.function(image=image, volumes={"/data": vol}, timeout=12 * 3600)
def train_owt():
    from cs336_basics.train_bpe_fast import train_bpe
    import time
    import pickle

    start = time.perf_counter()
    ans = train_bpe("/data/owt_train.txt", 32000, ["<|endoftext|>"])
    print(time.perf_counter() - start)
    with open("/data/tokenizer_owt.pkl", "wb") as f:
        pickle.dump(ans, f)
    vol.commit()


@app.function(image=image, volumes={"/data": vol}, cpu=4.0, memory=30 * 1024, timeout=1800)
def profile():
    from cs336_basics.train_bpe import train_bpe
    import pickle
    import cProfile

    with cProfile.Profile() as pr:
        ans = train_bpe("/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
        pr.dump_stats("/data/tokenizer_fast.prof")
    with open("/data/tokenizer.pkl", "wb") as f:
        pickle.dump(ans, f)
    vol.commit()
