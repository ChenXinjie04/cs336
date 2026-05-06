import time
import torch
import json


class Logger:
    def __init__(self, out):
        self.start_time = time.perf_counter()
        self.f = open(out, "w")

    def log_train(self, loss, step):
        if type(loss) is torch.Tensor:
            loss = loss.item()
        cur_time = time.perf_counter()
        total_time = cur_time - self.start_time
        print(f"train loss {loss}, step {step}, time {total_time}")
        d = {"train_loss": loss, "step": step, "time": total_time}
        s = json.dumps(d)
        self.f.write(s + "\n")
        self.f.flush()

    def log_valid(self, loss, step):
        if type(loss) is torch.Tensor:
            loss = loss.item()
        cur_time = time.perf_counter()
        total_time = cur_time - self.start_time
        print(f"valid loss {loss}, step {step}, time {cur_time - self.start_time}")
        d = {"valid_loss": loss, "step": step, "time": total_time}
        s = json.dumps(d)
        self.f.write(s + "\n")
        self.f.flush()
