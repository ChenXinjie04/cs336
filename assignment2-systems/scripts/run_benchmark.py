import itertools
import pandas as pd
from cs336_systems import benchmark


def run_benchmarks():
    fixed_params = {
        "vocab_size": 32000,
        "rope_theta": 10000,
        "device": "cuda",
        "warmup": 10,
        "n": 100,
    }

    # 按表格定义模型配置
    model_configs = [
        {
            "size": "small",
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        {
            "size": "medium",
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
        {
            "size": "large",
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        {
            "size": "xl",
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
        {
            "size": "10B",
            "d_model": 4608,
            "d_ff": 12288,
            "num_layers": 50,
            "num_heads": 36,
        },
    ]

    # 只扫运行时参数
    search_space = {
        "batch_size": [1, 2, 4, 8],
        "context_length": [512, 1024, 2048],
        "mode": ["forward", "forward_backward", "full"],
        "amp": [False, True],
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

            mean, std = benchmark(**benchmark_params)

            row = {
                "size": size,
                **benchmark_params,
                "mean": mean,
                "std": std,
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=[
            "size",
            "mode",
            "amp",
            "batch_size",
            "context_length",
        ]
    )

    return df


if __name__ == "__main__":
    df = run_benchmarks()

    print(df)

    df.to_csv("benchmark_results.csv", index=False)

    latex_table = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Benchmark results under different model and runtime configurations.",
        label="tab:benchmark-results",
    )

    with open("benchmark_results.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)
