import modal

app = modal.App("pytests")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("cs336_basics", remote_path="/root/proj/cs336_basics")
    .add_local_dir("tests", remote_path="/root/proj/tests")
)


@app.function(image=image, timeout=600)
def run_tests():
    import subprocess

    subprocess.run(
        ["python", "-m", "pytest", "tests/test_tokenizer.py", "-v"],
        cwd="/root/proj",
        check=False,
    )
