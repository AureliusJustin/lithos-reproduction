"""hp_infer.py — the latency-critical tenant of LithOS §8.1.

An inference service under Poisson load, reporting end-to-end request latency.
This is the job whose tail the paper's experiments protect: it must keep its
latency when a best-effort neighbour is placed on the same GPU.

Poisson arrivals matter and are not a detail. A closed loop (send, wait, send)
never queues, so it cannot show head-of-line blocking; an open loop at a fixed rate
does queue, but only in lockstep. Real services arrive at random, which is what
makes the tail sensitive to a neighbour's long kernels — and §7 says the paper's
clients "create Poisson loads".

    python3 bench/hp_infer.py <rate-per-sec> <seconds> [batch]
"""
import os, random, sys, time
import torch
import torch.nn as nn

# HP_MODEL selects the served model. "synthetic" is the small hand-written net the
# earlier numbers in docs/TECHNICAL_REPORT.md were measured with, kept as the default so
# those stay reproducible; the named models are the ones the paper actually serves
# (ResNet and friends, its Table 1/2 sets).
HP_MODEL = os.environ.get("HP_MODEL", "synthetic")


class Infer(nn.Module):
    """A small conv/GEMM inference model — short kernels, latency-sensitive."""
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, 256), nn.ReLU(), nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name, dev):
    """Returns (module, input tensor factory). Real models use 224x224, as served."""
    if name == "synthetic":
        return Infer().to(dev).eval(), (lambda b: torch.randn(b, 3, 64, 64, device=dev))
    from torchvision import models
    m = getattr(models, name)(weights=None).to(dev).eval()
    return m, (lambda b: torch.randn(b, 3, 224, 224, device=dev))


def main():
    rate  = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
    secs  = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    batch = int(sys.argv[3])   if len(sys.argv) > 3 else 8

    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda")
    model, mk_input = build_model(HP_MODEL, dev)
    x = mk_input(batch)

    with torch.no_grad():
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()

        lat, dropped = [], 0
        t_end = time.time() + secs
        next_arrival = time.time()
        while time.time() < t_end:
            now = time.time()
            if now < next_arrival:
                time.sleep(min(next_arrival - now, 0.002))
                continue
            # Queueing delay is part of the request's latency: a request that
            # arrived while the GPU was busy has already been waiting.
            arrived = next_arrival
            model(x)
            torch.cuda.synchronize()
            lat.append((time.time() - arrived) * 1e3)
            next_arrival += random.expovariate(rate)
            if next_arrival < time.time() - 1.0:      # fell a second behind
                next_arrival = time.time()
                dropped += 1

    lat.sort()
    n = len(lat)
    if n == 0:
        print("HP: no requests completed"); return
    print(f"HP model={HP_MODEL} rate={rate:.0f}/s n={n} tput={n/secs:.0f}/s "
          f"p50={lat[n//2]:.3f} p95={lat[int(n*0.95)]:.3f} p99={lat[int(n*0.99)]:.3f} ms "
          f"resets={dropped}")


if __name__ == "__main__":
    main()
