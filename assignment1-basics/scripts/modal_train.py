import modal

app = modal.App("train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("cs336_basics")
)
vol = modal.Volume.from_name("cs336-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, gpu="B200", timeout=300 * 60)
def train_tinystories():
    from cs336_basics.train_script import train

    train()
    vol.commit()
