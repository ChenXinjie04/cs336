import itertools
import pandas as pd
from cs336_systems import benchmark


def run_benchmarks():
    fixed_params = {
        "vocab_size": 32000,
        "rope_theta": 10000,
        "batch_size": 8,
        "device": "cuda",
        "warmup": 10,
        "n": 100,
    }

    # 按表格定义模型配置
    model_configs = [
        {
            "size": "medium",
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
    ]

    # 只扫运行时参数
    search_space = {
        "d_model": [16, 32, 64, 128],
        "context_length": [256, 1024, 4096, 8192, 16384],
    }

    rows = []

    keys = list(search_space.keys())
    values = list(search_space.values())

    for model_config in model_configs:
        for combo in itertools.product(*values):
            runtime_params = dict(zip(keys, combo))

            all_params = {
                **fixed_params,
                **model_config,
                **runtime_params,
            }

            print(f"Running: {all_params}")

            # 注意：如果 benchmark 不接受 size 参数，要把它去掉
            benchmark_params = all_params.copy()
            size = benchmark_params.pop("size")

            benchmark(**benchmark_params)


if __name__ == "__main__":
    df = run_benchmarks()
