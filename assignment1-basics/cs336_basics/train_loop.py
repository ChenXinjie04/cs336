import numpy as np
import torch


def data_loading(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    input = np.stack([dataset[i : i + context_length] for i in starts]).astype(np.int64)
    target = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)

    input = torch.from_numpy(input).to(device)
    target = torch.from_numpy(target).to(device)
    return (input, target)


def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save({"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}, out)


def load_checkpoint(src, model, optimizer, device):
    ckpt = torch.load(src, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["iteration"] + 1
