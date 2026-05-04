import torch


def cross_entropy(pred, targets):
    c = torch.max(pred, keepdim=True, dim=-1).values
    pred -= c
    actual = torch.gather(pred, -1, targets.unsqueeze(-1))
    pred = torch.exp(pred)
    pred = torch.sum(pred, dim=-1)
    return (torch.log(pred) - actual).mean()
