import json
import matplotlib.pyplot as plt


def load_train(src):
    steps, loss, lrs = [], [], []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "train_loss" in d:
                steps.append(d["step"])
                loss.append(d["train_loss"])
                lrs.append(d["lr"])
    return steps, loss, lrs


def plot_lr_range_test(lrs, loss, smooth_beta=0.0, save_path=None):
    if smooth_beta > 0:
        smoothed, avg = [], 0
        for i, l in enumerate(loss):
            avg = smooth_beta * avg + (1 - smooth_beta) * l
            # bias correction
            smoothed.append(avg / (1 - smooth_beta ** (i + 1)))
        loss_to_plot = smoothed
    else:
        loss_to_plot = loss

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, loss_to_plot, linewidth=1.5)
    ax.set_xscale("log")  # 关键：x 轴对数刻度
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss" + (" (smoothed)" if smooth_beta > 0 else ""))
    ax.set_title("LR Range Test")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    min_loss = min(loss_to_plot)
    ax.set_ylim(min_loss * 0.9, min_loss * 4)

    min_idx = loss_to_plot.index(min(loss_to_plot))
    ax.axvline(lrs[min_idx], color="red", linestyle=":", alpha=0.6, label=f"min loss @ lr={lrs[min_idx]:.2e}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


steps, loss, lr_vals = load_train("./data/log.jsonl")
plot_lr_range_test(lr_vals, loss, smooth_beta=0.9, save_path="./lr_range_test.png")
