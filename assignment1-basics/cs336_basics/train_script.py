from torch.nn import SoftMarginLoss
from cs336_basics.train_loop import data_loading, save_checkpoint
from cs336_basics.train import cross_entropy, AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.logger import Logger
import numpy as np
import torch


DATA_PATH = "/data/tinystories_tokens.npy"
VALID_DATA_PATH = "/data/tinystories_valid_tokens.npy"
CKPT_PATH = "/data/tinystories_ckpt.pt"
LOG_PATH = "/data/log.jsonl"
SAVE_RATE = 500
VALID_RATE = 50
batch_size = 3
context_length = 1024
vocab_size = 3000
num_layers = 12
d_model = 1600
d_ff = 4888
theta = 10000
num_heads = 25
dtype = torch.float
device = "cuda"
max_step = 5000
lr = 0.001
weight_decay = 0.001
betas = (0.99, 0.999)
max_learning_rate = 0.001
min_learning_rate = 0.0001
warmup_iters = 1000
cosin_cycle_iters = 4000
max_l2_norm = 10
model = TransformerLM(vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta, device, dtype)
optimizer = AdamW(model.parameters(), lr, weight_decay, betas, 0.0001)
train_data = np.memmap(DATA_PATH, np.int64)
valid_data = np.memmap(VALID_DATA_PATH, np.int64)


def train():
    logger = Logger(LOG_PATH)
    for step in range(max_step):
        input, target = data_loading(train_data, batch_size, context_length, device)
        output = model.forward(input)
        loss = cross_entropy(output, target)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)
        for g in optimizer.param_groups:
            g["lr"] = lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosin_cycle_iters)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logger.log_train(loss, step)
        if step % VALID_RATE == 0:
            input, target = data_loading(valid_data, batch_size, context_length, device)
            with torch.no_grad():
                output = model.forward(input)
                valid_loss = cross_entropy(output, target)
            logger.log_valid(valid_loss, step)
        if step % SAVE_RATE == 0:
            save_checkpoint(model, optimizer, step, CKPT_PATH)


def sample_next_token(input, temperature, top_p):
    input = torch.tensor([input], dtype=torch.int64)
    output = model.forward(input)
    q = output[0, -1]
    q /= temperature
    q = softmax(q, dim=-1)
    sorted_value, sorted_index = torch.sort(q, descending=True)
    i = 0
    sum = 0
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
    length = 0
    while length < max_new_tokens:
        new_token_id = sample_next_token(input, temperature, top_p)
        if new_token_id == eos_token_id:
            break
        input.append(new_token_id)
        length += 1
    return input
