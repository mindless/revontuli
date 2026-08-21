#!/usr/bin/env python3
"""I test whether PyTorch MPS is trustworthy on this Intel Mac + RX 6900 XT.

Two upstream reports drive this:
  - pytorch/pytorch#141864  BFloat16 is not supported on MPS (Intel + AMD)
  - pytorch/pytorch#178697  F.linear is catastrophically wrong on RDNA2 when
                            in_features >= 256, while torch.mm is fine

Both are closed as "not planned", so I verify them on THIS machine rather than
trusting the tracker. If F.linear is wrong here, PyTorch MPS is unusable as a
MiniMax H3 backend, because every attention and MLP projection in H3 is a
linear layer with in_features in the thousands.
"""

import torch
import torch.nn.functional as F

print(f"torch            = {torch.__version__}")
print(f"mps available    = {torch.backends.mps.is_available()}")
print(f"mps built        = {torch.backends.mps.is_built()}")
print()

if not torch.backends.mps.is_available():
    print("MPS is unavailable; nothing to test.")
    raise SystemExit(0)

dev = torch.device("mps")
torch.manual_seed(0)

# ---------------------------------------------------------------- BF16 support
print("=== BFloat16 support on MPS ===")
try:
    t = torch.randn(64, 64, dtype=torch.bfloat16, device=dev)
    r = t @ t
    torch.mps.synchronize()
    print(f"  BF16 matmul: OK (result dtype {r.dtype}, "
          f"finite={bool(torch.isfinite(r.float()).all())})")
    bf16_ok = True
except Exception as exc:                                    # noqa: BLE001
    print(f"  BF16 matmul: FAILED -> {type(exc).__name__}: {exc}")
    bf16_ok = False
print()

# ------------------------------------------------- F.linear vs torch.mm vs CPU
print("=== F.linear vs torch.mm, across in_features (issue #178697) ===")
print(f"  {'in_feat':>8} {'dtype':>8} {'mm max_err':>13} "
      f"{'linear max_err':>15}  verdict")
print("  " + "-" * 62)

linear_broken = []
for dtype in (torch.float32, torch.float16):
    for in_features in (64, 128, 255, 256, 512, 1024, 3072):
        out_features, batch = 128, 32
        x_c = torch.randn(batch, in_features, dtype=dtype)
        w_c = torch.randn(out_features, in_features, dtype=dtype)

        # CPU reference computed in float64 for a clean baseline.
        ref = (x_c.double() @ w_c.double().T)

        x_m, w_m = x_c.to(dev), w_c.to(dev)

        mm = (x_m @ w_m.T).cpu().double()
        lin = F.linear(x_m, w_m).cpu().double()
        torch.mps.synchronize()

        scale = ref.abs().max().item() or 1.0
        mm_err = (mm - ref).abs().max().item() / scale
        lin_err = (lin - ref).abs().max().item() / scale

        tol = 1e-5 if dtype is torch.float32 else 5e-3
        bad = lin_err > max(tol * 50, 1e-2)
        if bad:
            linear_broken.append((str(dtype), in_features, lin_err))
        print(f"  {in_features:>8} {str(dtype).replace('torch.',''):>8} "
              f"{mm_err:>13.3e} {lin_err:>15.3e}  "
              f"{'*** F.linear WRONG ***' if bad else 'ok'}")

print()
print("=== Verdict ===")
if bf16_ok:
    print("  BF16 on MPS: supported on this machine.")
else:
    print("  BF16 on MPS: NOT supported -> h3.c's BF16 weights could not be")
    print("               fed to a PyTorch MPS backend without conversion.")
if linear_broken:
    print(f"  F.linear: REPRODUCED the RDNA2 defect in {len(linear_broken)} "
          f"configuration(s).")
    print("  I therefore mark PyTorch MPS UNSAFE as a MiniMax H3 backend on")
    print("  this machine. Every H3 projection uses in_features in the")
    print("  thousands, which is squarely inside the broken range.")
else:
    print("  F.linear: I could NOT reproduce the defect on this machine.")
print()
print(f"  Note: the newest PyTorch wheel for macOS x86_64 is 2.2.2; upstream")
print(f"  stopped shipping Intel-Mac builds after it. Installed: "
      f"{torch.__version__}")
