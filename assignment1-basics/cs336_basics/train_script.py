from cs336_basics.train_loop import data_loading, load_checkpoint, save_checkpoint
from cs336_basics.train import cross_entropy, AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.logger import Logger
import numpy as np
import torch
import os


lr = 5e-3
DATA_PATH = "/data/tinystories_tokens.npy"
VALID_DATA_PATH = "/data/tinystories_valid_tokens.npy"
CKPT_PATH = "/data/tinystories_ckpt.pt"
LOG_PATH = "/data/log.jsonl"
SAVE_RATE = 500
VALID_RATE = 50
batch_size = 128
max_step = 10000
context_length = 256
vocab_size = 10000
num_layers = 4
d_model = 512
d_ff = 1344
theta = 10000
num_heads = 16
dtype = torch.bfloat16
device = "cuda"
weight_decay = 0.1
betas = (0.9, 0.95)
max_learning_rate = lr
min_learning_rate = lr * 0.1
warmup_iters = max_step * 0.01
cosin_cycle_iters = max_step - warmup_iters
max_l2_norm = 1
model = TransformerLM(vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta, device, dtype)
optimizer = AdamW(model.parameters(), lr, weight_decay, betas, 0.0001)


def train():
    train_data = np.memmap(VALID_DATA_PATH, np.uint16)
    valid_data = np.memmap(VALID_DATA_PATH, np.uint16)
    logger = Logger(LOG_PATH)
    start_step = 0
    lr = 0
    if os.path.exists(CKPT_PATH):
        start_step = load_checkpoint(CKPT_PATH, model, optimizer, device)
    for step in range(start_step, max_step):
        input, target = data_loading(train_data, batch_size, context_length, device)
        output = model.forward(input)
        loss = cross_entropy(output, target)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)
        for g in optimizer.param_groups:
            lr = lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosin_cycle_iters)
            g["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logger.log_train(loss, step, lr)
        if step % VALID_RATE == 0:
            input, target = data_loading(valid_data, batch_size, context_length, device)
            with torch.no_grad():
                output = model.forward(input)
                valid_loss = cross_entropy(output, target)
            logger.log_valid(valid_loss, step, lr)
        if step % SAVE_RATE == 0:
            print(f"{torch.cuda.max_memory_allocated() / 1e9 =}")
            save_checkpoint(model, optimizer, step, CKPT_PATH)
    save_checkpoint(model, optimizer, max_step - 1, CKPT_PATH)


def sample_next_token(input, temperature, top_p):
    assert top_p <= 1
    input = torch.tensor([input], dtype=torch.int64)
    output = model.forward(input)
    q = output[0, -1].float()
    q /= temperature
    q = softmax(q, dim=-1)
    sorted_value, sorted_index = torch.sort(q, descending=True)
    i = 0
    sum = 0.0
    while sum < top_p:
        sum += sorted_value[i]
        i += 1
    sorted_index = sorted_index[:i]
    sorted_value = sorted_value[:i]
    sum = sorted_value.sum()
    sorted_value /= sum
    idx = torch.multinomial(sorted_value, num_samples=1)
    return sorted_index[idx].item()


def decode(input, temperature, max_new_tokens, eos_token_id, top_p=0.9):
    load_checkpoint("./data/tinystories_ckpt_lr1e-03.pt", model, optimizer, device)
    length = len(input)
    while length < max_new_tokens:
        new_token_id = sample_next_token(input, temperature, top_p)
        if new_token_id == eos_token_id:
            break
        input.append(new_token_id)
        length += 1
    return input


if __name__ == "__main__":
    train()
