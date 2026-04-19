import modal

app = modal.App("example")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .pip_install("regex")
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
def download_data():
    import subprocess

    subprocess.run(
        "wget -O /data/TinyStoriesV2-GPT4-train.txt --progress=dot:giga https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
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
def train():
    from cs336_basics.tokenization import train_bpe
    import pickle
    import time

    start = time.perf_counter()
    ans = train_bpe("/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    with open("/data/tokenizer.pkl", "wb") as f:
        pickle.dump(ans, f)
    vol.commit()
    print(time.perf_counter() - start)
