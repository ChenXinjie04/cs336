from collections.abc import Callable
import torch
import math


def cross_entropy(pred, targets):
    c = torch.max(pred, keepdim=True, dim=-1).values
    pred -= c
    actual = torch.gather(pred, -1, targets.unsqueeze(-1))
    pred = torch.exp(pred)
    pred = torch.sum(pred, dim=-1)
    return (torch.log(pred) - actual).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        default = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, default)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m: float = state.get("m", 0)
                v: float = state.get("v", 0)
                grad = p.grad.data
                alpha = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data -= alpha * m / (v**0.5 + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 1 / 2 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm):
    total_grad = 0
    eps = 1e-6
    for p in parameters:
        if p.grad is None:
            continue
        total_grad += (p.grad * p.grad).sum()
    total_grad = total_grad**0.5
    if total_grad < max_l2_norm:
        return
    factor = max_l2_norm / (total_grad + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad *= factor
