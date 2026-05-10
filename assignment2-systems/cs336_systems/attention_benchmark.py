from cs336_basics.model import scaled_dot_product_attention
from itertools import product
import torch
import timeit
import statistics
import pandas as pd


def benchmark(batch_size, d_model, context_length, device, warmup, n):
    def make_tensor():
        x = torch.empty(batch_size, context_length, d_model, device=device)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=1.0, a=-3, b=3)
        return x.requires_grad_(True)

    q = make_tensor()
    k = make_tensor()
    v = make_tensor()
    mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool, device=device))

    for _ in range(warmup):
        output = scaled_dot_product_attention(q, k, v, mask)
        output.backward(torch.ones_like(output))

    forward, backward = [], []
    mem_before_backward = 0
    max_before_backward = 0
    for _ in range(n):
        start = timeit.default_timer()
        output = scaled_dot_product_attention(q, k, v, mask)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward.append(end - start)
        mem_before_backward = torch.cuda.memory_allocated()
        max_before_backward = torch.cuda.max_memory_allocated()
        start = timeit.default_timer()
        output.backward(torch.ones_like(output))
        torch.cuda.synchronize()
        end = timeit.default_timer()
        backward.append(end - start)

    return {
        "forward_mean": statistics.mean(forward),
        "forward_std": statistics.stdev(forward),
        "backward_mean": statistics.mean(backward),
        "backward_std": statistics.stdev(backward),
        "mem_before_backward": mem_before_backward / 1024**3,
        "max_before_backward": max_before_backward / 1024**3,
    }


if __name__ == "__main__":
    d_models = [16, 32, 64, 128]
    context_lengths = [256, 1024, 4096, 8192, 16384]

    configs = [
        {
            "batch_size": 8,
            "device": "cuda",
            "warmup": 10,
            "n": 100,
            "d_model": d_model,
            "context_length": context_length,
        }
        for d_model, context_length in product(d_models, context_lengths)
    ]

    rows = []
    for config in configs:
        try:
            print("running: ", config)
            rows.append({**config, **benchmark(**config)})
        except torch.cuda.OutOfMemoryError:
            print("OOM: ", config)
            rows.append(
                {
                    **{
                        "forward_mean": "oom",
                        "forward_std": "oom",
                        "backward_mean": "oom",
                        "backward_std": "oom",
                    },
                    **config,
                }
            )
        finally:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, float_format="%.6f")
    with open("result.tex", "w") as f:
        f.write(latex)
