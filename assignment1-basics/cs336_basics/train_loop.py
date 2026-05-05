import numpy as np
import torch


def data_loading(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    input = np.stack([dataset[i : i + context_length] for i in starts]).astype(np.int64)
    target = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)

    input = torch.from_numpy(input).to(device)
    target = torch.from_numpy(target).to(device)
    return (input, target)
