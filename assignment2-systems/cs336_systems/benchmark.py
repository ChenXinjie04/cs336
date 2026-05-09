from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import statistics
import torch
import argparse
import timeit


def benchmark(vocab_size, batch_size, d_model, d_ff, num_layers, num_heads, context_length, rope_theta, device, warmup, n, mode):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    optimizer = AdamW(model.parameters())

    def forward():
        model.forward(x)
        torch.cuda.synchronize()

    def forward_backward():
        output = model.forward(x)
        loss = cross_entropy(output, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()

    def full():
        output = model.forward(x)
        loss = cross_entropy(output, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    def run(fn):
        times = []
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(n):
            times.append(timeit.timeit(fn, number=1))
        return times

    if mode == "forward":
        times = run(forward)
    elif mode == "forward_backward":
        times = run(forward_backward)
    else:
        times = run(full)

    mean = statistics.mean(times)
    std = statistics.stdev(times)
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocab size")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--d_model", type=int, default=768, help="d_model")
    parser.add_argument("--d_ff", type=int, default=3072, help="d_ff")
    parser.add_argument("--num_layers", type=int, default=12, help="num_layers")
    parser.add_argument("--num_heads", type=int, default=12, help="num_heads")
    parser.add_argument("--context_length", type=int, default=256, help="context length")
    parser.add_argument("--rope_theta", type=int, default=10000, help="rope theta")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--warmup", type=int, default=10, help="warmup")
    parser.add_argument("--n", type=int, default=100, help="n")
    parser.add_argument(
        "--mode", type=str, default="forward", choices=["forward", "forward_backward", "full"], help="forward only, forward+backward or full training step with optimizer"
    )

    args = parser.parse_args()
    mean, std = benchmark(
        args.vocab_size,
        args.batch_size,
        args.d_model,
        args.d_ff,
        args.num_layers,
        args.num_heads,
        args.context_length,
        args.rope_theta,
        args.device,
        args.warmup,
        args.n,
        args.mode,
    )
    print(mean)
    print(std)
