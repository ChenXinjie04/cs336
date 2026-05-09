import modal

app = modal.App("train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .run_commands(
        "wget -O /tmp/nsys.deb https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb",
        "apt install -y /tmp/nsys.deb",
    )
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install_from_pyproject("./cs336-basics/pyproject.toml")
    .add_local_python_source("cs336_systems")
    .add_local_python_source("cs336_basics")
)
vol = modal.Volume.from_name("cs336-data", create_if_missing=True)


@app.function(image=image, gpu="A100-40GB", timeout=300 * 60)
def benchmark(config):
    from cs336_systems.benchmark import benchmark

    size = config.pop("size")
    mode = config["mode"]
    mean, std = benchmark(**config)
    return (size, mode, mean, std)


def prepare_helper():
    common = {"vocab_size": 10000, "batch_size": 4, "context_length": 512, "rope_theta": 10000, "device": "cuda", "warmup": 2, "n": 10}
    configs = [
        {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "mode": "forward", **common},
        {"size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "mode": "forward", **common},
        {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, "mode": "forward", **common},
        {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "mode": "forward_backward", **common},
        {"size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "mode": "forward_backward", **common},
        {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, "mode": "forward_backward", **common},
        {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "mode": "full", **common},
        {"size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "mode": "full", **common},
        {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, "mode": "full", **common},
        # {"size": "xl", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32, **common},
    ]
    results = list(benchmark.map(configs))
    for size, mode, mean, std in results:
        print(f"{size} | {mode} | {mean} | {std}")


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol})
def run_profile():
    import subprocess

    result = subprocess.run(
        ["nsys", "profile", "-o", "/data/report", "--trace=cuda,nvtx", "--force-overwrite=true", "--", "python", "-m", "cs336_systems.benchmark"],
        # ["nvidia-smi"],
        # ["python", "-m", "cs336_systems.benchmark"],
        # ["nsys", "status", "-e"],
        capture_output=True,
    )
    print("STDERR: ", result.stderr)
    print("STDOUT: ", result.stdout)
    vol.commit()


@app.local_entrypoint()
def main():
    run_profile.remote()
