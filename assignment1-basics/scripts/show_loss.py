import json
import numpy as np
import matplotlib.pyplot as plt


def load_log(src):
    train_steps, train_loss, lrs = [], [], []
    valid_steps, valid_loss = [], []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "train_loss" in d:
                train_steps.append(d["step"])
                train_loss.append(d["train_loss"])
                lrs.append(d["lr"])
            elif "valid_loss" in d:
                valid_steps.append(d["step"])
                valid_loss.append(d["valid_loss"])
    return train_steps, train_loss, lrs, valid_steps, valid_loss


def smooth(y, window=51):
    """简单的滑动平均平滑"""
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    # 用 'same' 模式保持长度，边缘用 edge padding 减少失真
    pad = window // 2
    y_padded = np.pad(y, pad, mode="edge")
    return np.convolve(y_padded, kernel, mode="valid")


train_steps, train_loss, lr_vals, valid_steps, valid_loss = load_log("./data/log.jsonl")

# 取最大 lr 作为图例标签（cosine/warmup 调度下的峰值 lr）
peak_lr = max(lr_vals) if lr_vals else 0.0
lr_label = f"lr={peak_lr:.0e}"

train_loss_smooth = smooth(train_loss, window=51)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Train loss: 原始数据淡色 + 平滑后深色
ax1.plot(train_steps, train_loss, color="tab:blue", alpha=0.25, linewidth=0.8)
ax1.plot(train_steps, train_loss_smooth, color="tab:blue", linewidth=1.8, label=lr_label)
ax1.set_yscale("log")
ax1.set_ylabel("train loss")
ax1.set_title("Train loss vs learning rate")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# Valid loss
ax2.plot(valid_steps, valid_loss, color="tab:blue", linewidth=1.2)
ax2.set_yscale("log")
ax2.set_ylabel("valid loss")
ax2.set_xlabel("step")
ax2.set_title("Valid loss vs learning rate")
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
