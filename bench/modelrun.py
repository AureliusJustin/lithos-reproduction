"""modelrun.py — run one model for a fixed number of iterations, so LithOS's
predictor sees the same operator sequence every time.

The scaling-curve figures (paper Figs 12, 13) need per-kernel latency for a whole
workload measured under several TPC counts and several GPU clocks. That means a
harness that is (a) fixed-shape, so operator ordinals line up across runs, and
(b) one model per invocation, so each panel is a real workload rather than a mix.

    python3 bench/modelrun.py <model> <infer|train> <batch> <iters>
"""
import sys, time
import torch
import torch.nn as nn


def main():
    name  = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
    mode  = sys.argv[2] if len(sys.argv) > 2 else "infer"
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    torch.manual_seed(0)
    dev = torch.device("cuda")
    from torchvision import models
    model = getattr(models, name)(weights=None).to(dev)
    x = torch.randn(batch, 3, 224, 224, device=dev)

    if mode == "train":
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)
        y = torch.randint(0, 1000, (batch,), device=dev)
        loss_fn = nn.CrossEntropyLoss()
        step = lambda: (opt.zero_grad(set_to_none=True),
                        loss_fn(model(x), y).backward(), opt.step())
        ctx = torch.enable_grad()
    else:
        model.eval()
        step = lambda: model(x)
        ctx = torch.no_grad()

    with ctx:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            step()
            torch.cuda.synchronize()        # batch boundary, per §5.7
        el = time.time() - t0
    print(f"modelrun {name} {mode} batch={batch} iters={iters} "
          f"{el/iters*1e3:.3f} ms/iter")


if __name__ == "__main__":
    main()
