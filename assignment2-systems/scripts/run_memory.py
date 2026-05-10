import itertools
from cs336_systems.memory_profiling import benchmark


def make_snap_path(params):
    """
    根据当前 benchmark 配置生成 snapshot 文件名
    """
    return (
        f"memory_snapshot_"
        f"{params['size']}_"
        f"ctx{params['context_length']}_"
        f"{params['mode']}_"
        f"amp{int(params['amp'])}_"
        f"bs{params['batch_size']}_"
        f"d{params['d_model']}_"
        f"layers{params['num_layers']}_"
        f"heads{params['num_heads']}"
        f".pickle"
    )


def run_benchmarks():
    fixed_params = {"vocab_size": 32000, "rope_theta": 10000, "device": "cuda", "batch_size": 4}

    # 按表格定义模型配置
    model_configs = [
        {
            "size": "xl",
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
    ]

    # 只扫运行时参数
    search_space = {
        "context_length": [128, 2048],
        "mode": ["forward", "full"],
        "amp": [False, True],
    }

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
            all_params["snap_path"] = make_snap_path(all_params)

            # 注意：如果 benchmark 不接受 size 参数，要把它去掉
            benchmark_params = all_params.copy()
            benchmark_params.pop("size")

            benchmark(**benchmark_params)


if __name__ == "__main__":
    run_benchmarks()
