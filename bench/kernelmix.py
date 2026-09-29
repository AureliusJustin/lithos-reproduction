"""kernelmix.py — a real DL kernel mix, for the two studies a synthetic loop cannot do.

Two claims in this repo were measured on hand-written kernels and were weaker for
it:

  * right-sizing accuracy (§8.2) was fitted on a single tiled matmul, the most
    Amdahl-like kernel available, where R^2 is trivially near 1. The paper's
    0.92-0.99 is an execution-time-weighted average across ALL kernels of real
    workloads, including the short-runtime outliers that motivated its occupancy
    filter in the first place.
  * DVFS (§5.6) was measured on kernels that all turned out to be clock-bound
    (sensitivity >= 0.8). The model's value is largest at LOW sensitivity, and that
    end went unverified.

Both need a workload whose kernels genuinely differ. This model is chosen for that
spread rather than for accuracy on any task: convolutions and GEMMs (compute-bound,
cuDNN/cuBLAS), normalisations and softmax (reduction-heavy), and large elementwise
ops (bandwidth-bound). Everything is fixed-shape and deterministic, and each
iteration ends in a synchronize so LithOS's operator ordinals (§5.7) line up
identically across runs — which is what lets measurements from separate runs at
different TPC counts or clocks be joined per operator.

    python3 bench/kernelmix.py [iters]
"""
import sys, time
import torch
import torch.nn as nn


class Mixed(nn.Module):
    """Deliberately heterogeneous: each block stresses a different limit."""
    def __init__(self, ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
        self.fc1   = nn.Linear(ch, 1024)
        self.ln    = nn.LayerNorm(1024)
        self.fc2   = nn.Linear(1024, 512)

    def forward(self, x, big):
        x = torch.relu(self.bn1(self.conv1(x)))       # conv + norm
        x = torch.relu(self.bn2(self.conv2(x)))       # conv + norm
        p = x.mean(dim=(2, 3))                        # spatial reduction
        h = torch.relu(self.fc1(p))                   # GEMM
        h = self.ln(h)                                # reduction + elementwise
        h = torch.softmax(self.fc2(h), dim=-1)        # GEMM + softmax
        # A bandwidth-bound tail: no reuse, working set far past the L2, so its
        # latency is set by memory rather than by the SM clock.
        big = big * 1.0001 + 0.5
        return h.sum() + big.sum()


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    torch.manual_seed(0)
    dev = torch.device("cuda")
    model = Mixed().to(dev).eval()
    x   = torch.randn(32, 3, 64, 64, device=dev)
    big = torch.randn(64 * 1024 * 1024 // 4, device=dev)   # 64 MB, > A100 L2

    with torch.no_grad():
        for _ in range(10):                     # warm up: autotuning, allocator
            model(x, big)
        torch.cuda.synchronize()

        t0 = time.time()
        out, lat = None, []
        for _ in range(iters):
            a0 = time.time()
            out = model(x, big)
            torch.cuda.synchronize()            # a batch boundary, per §5.7
            lat.append((time.time() - a0) * 1e3)
        el = time.time() - t0

    lat.sort()
    print(f"kernelmix: {iters} iters in {el:.3f} s "
          f"p50={lat[len(lat)//2]:.3f} p99={lat[int(len(lat)*0.99)]:.3f} ms "
          f"checksum={float(out):.4f}")


if __name__ == "__main__":
    main()
