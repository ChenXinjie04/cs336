import modal

app = modal.App("pytests")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("cs336_basics")
)
vol = modal.Volume.from_name("cs336-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=1200)
def tokenize_tinystories():
    import numpy as np
    from cs336_basics.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_files("/data/tokenizer_tiny_stories.pkl")
    with open("/data/TinyStoriesV2-GPT4-train.txt", encoding="utf-8") as f:
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save("/data/tinystories_tokens.npy", arr)
    vol.commit()


@app.function(image=image, volumes={"/data": vol}, timeout=7200)
def tokenize_owt():
    import numpy as np
    from cs336_basics.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_files("/data/tokenizer_owt.pkl")
    with open("/data/owt_train.txt", encoding="utf-8") as f:
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save("/data/owt_tokens.npy", arr)
    vol.commit()
