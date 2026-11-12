"""Exhaustive check: uint8 -> float32 image normalization (x.to(f32) / 255.0 * 2.0 - 1.0) on CPU vs CUDA.
Every uint8 value at many positions/lengths (vectorized bodies and scalar tails, odd offsets)."""
import torch

def f(x):
    return x.to(torch.float32) / 255.0 * 2.0 - 1.0

ok = True
vals = torch.arange(256, dtype=torch.uint8)
for n in [1, 3, 7, 16, 31, 255, 256, 257, 1000, 4097, 224 * 224 * 3, 128 * 224 * 224 * 3 + 5]:
    for shift in [0, 1, 13]:
        x = vals.repeat((n + 255) // 256 + 1)[shift:shift + n].clone()
        a = f(x)
        b = f(x.cuda()).cpu()
        if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
            ok = False
            bad = (a.view(torch.int32) != b.view(torch.int32)).nonzero()[:5].flatten()
            print("DIFF n", n, "shift", shift, "at", bad.tolist(), x[bad].tolist(), a[bad].tolist(), b[bad].tolist())
# non-contiguous / 4-D HWC images, like a batch of camera frames
x = torch.randint(0, 256, (8, 224, 224, 3), dtype=torch.uint8)
for t in [x, x.permute(0, 3, 1, 2), x[:, ::2]]:
    if not torch.equal(f(t).contiguous().view(torch.int32), f(t.cuda()).cpu().contiguous().view(torch.int32)):
        ok = False
        print("DIFF layout", t.shape, t.stride())
print("torch", torch.__version__, "EXHAUSTIVE", "PASS" if ok else "FAIL")
