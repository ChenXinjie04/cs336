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
    for _ in range(n):
        start = timeit.default_timer()
        output = scaled_dot_product_attention(q, k, v, mask)
        end = timeit.default_timer()
        forward.append(end - start)

        start = timeit.default_timer()
        output.backward(torch.ones_like(output))
        end = timeit.default_timer()
        backward.append(end - start)

    return {
        "forward_mean": statistics.mean(forward),
        "forward_std": statistics.stdev(forward),
        "backward_mean": statistics.mean(backward),
        "backward_std": statistics.stdev(backward),
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
        print("running ", config)
        rows.append({**benchmark(**config), **config})

    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, float_format="%.6f")
    with open("result.tex", "w") as f:
        f.write(latex)
