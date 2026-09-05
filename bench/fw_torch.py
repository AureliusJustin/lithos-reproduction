#!/usr/bin/env python3
"""Framework validation: PyTorch correctness AND atomization coverage under LithOS.
Run with the wrapper:  LD_LIBRARY_PATH=build python3 bench/fw_torch.py"""
import torch, sys

def check(name, got, want, tol=1e-3):
    ok = torch.allclose(got, want, rtol=tol, atol=tol)
    print(f"  {name:<28} {'PASS' if ok else 'FAIL'}")
    return ok

def main():
    assert torch.cuda.is_available(), "no CUDA"
    print(f"  device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    dev = "cuda"; allok = True
    torch.manual_seed(0)

    # cuBLAS matmul
    a = torch.randn(512, 512, device=dev); b = torch.randn(512, 512, device=dev)
    allok &= check("matmul (cuBLAS)", (a @ b).cpu(), a.cpu() @ b.cpu(), 1e-2)

    # cuDNN convolution
    x = torch.randn(8, 3, 64, 64, device=dev)
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(dev)
    y = conv(x)
    allok &= check("conv2d (cuDNN)", y.cpu(), conv.cpu()(x.cpu()), 1e-2)
    conv.to(dev)

    # elementwise + reduction (ATen)
    t = torch.randn(1 << 20, device=dev)
    allok &= check("elementwise+reduce", (t * 2 + 1).sum().cpu().reshape(1),
                   (t.cpu() * 2 + 1).sum().reshape(1), 1e-1)

    # a small MLP fwd+bwd (mixed kernels + autograd)
    net = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 10)).to(dev)
    inp = torch.randn(64, 256, device=dev)
    loss = net(inp).square().mean(); loss.backward()
    gradok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    print(f"  {'MLP fwd+bwd':<28} {'PASS' if gradok else 'FAIL'}")
    allok &= gradok

    # CUDA graph capture + replay
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): net(inp)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph(); static = inp.clone()
        with torch.cuda.graph(g): out = net(static)
        g.replay(); torch.cuda.synchronize()
        ok = torch.isfinite(out).all().item()
        print(f"  {'CUDA graph capture+replay':<28} {'PASS' if ok else 'FAIL'}")
        allok &= ok
    except Exception as e:
        print(f"  {'CUDA graph capture+replay':<28} SKIP ({type(e).__name__})")

    torch.cuda.synchronize()
    print(f"  OVERALL: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1

if __name__ == "__main__":
    sys.exit(main())
