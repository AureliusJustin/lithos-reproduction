"""be_train.py — the best-effort tenant of LithOS §8.1.

A training loop: forward, backward, step, in a closed loop with no deadline. The
paper's point is that training produces LONG kernels — it measures p99 kernel
latencies above 30 ms for DLRM — and that those kernels are what block a
co-located latency-critical service, because a GPU cannot preempt one.

Batch size is the knob: §8.4 varies it to show head-of-line blocking growing with
kernel duration, and it is the same sweep here.

    python3 bench/be_train.py <seconds> [batch] [width]
"""
import os, sys, time
import torch
import torch.nn as nn

# BE_MODEL selects the training job. "synthetic" is the wide MLP the earlier
# numbers in docs/TECHNICAL_REPORT.md were measured with, kept as the default; the named
# models are from the paper's Table 1 training set (ResNet-50, VGG-19,
# MobileNetV2). With a real model the `width` argument is ignored and `batch` is
# the training batch size, which is the knob §8.4 sweeps.
BE_MODEL = os.environ.get("BE_MODEL", "synthetic")


class Train(nn.Module):
    """Wide GEMMs: long kernels, the noisy neighbour of the experiment."""
    def __init__(self, width=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 512),
        )

    def forward(self, x):
        return self.net(x)


def main():
    secs  = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    batch = int(sys.argv[2])   if len(sys.argv) > 2 else 4096
    width = int(sys.argv[3])   if len(sys.argv) > 3 else 4096

    torch.manual_seed(0)
    dev = torch.device("cuda")
    if BE_MODEL == "synthetic":
        model = Train(width).to(dev)
        x = torch.randn(batch, width, device=dev)
        y = torch.randn(batch, 512, device=dev)
        loss_fn = nn.MSELoss()
    else:
        from torchvision import models
        model = getattr(models, BE_MODEL)(weights=None).to(dev)
        x = torch.randn(batch, 3, 224, 224, device=dev)
        y = torch.randint(0, 1000, (batch,), device=dev)
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for _ in range(3):                      # warm up allocator + autotuning
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()
    torch.cuda.synchronize()

    n, t0, t_end = 0, time.time(), time.time() + secs
    while time.time() < t_end:
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()
        torch.cuda.synchronize()
        n += 1
    el = time.time() - t0
    print(f"BE model={BE_MODEL} batch={batch} width={width} iters={n} tput={n/el:.2f}/s")


if __name__ == "__main__":
    main()
