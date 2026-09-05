"""Triton: JIT-compiles its own kernels and loads them through the driver API,
a different module path from the vendor libraries."""
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

n = 1 << 20
x = torch.rand(n, device='cuda'); y = torch.rand(n, device='cuda')
o = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
torch.cuda.synchronize()
ok = torch.allclose(o, x + y)
print(f"  triton vector_add       {'PASS' if ok else 'FAIL'}")
print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
