"""A Python project that calls a LYTH kernel: torch for memory, the generated module to launch.

    cargo run -p lyth -- build examples/saxpy.lyth --machine <your machine file> -o saxpy.ptx --bind-py saxpy.py
    python use_saxpy.py
"""
import torch

import saxpy

n = 1 << 20
x = torch.arange(n, device="cuda", dtype=torch.float32) * 0.001
y = 1.0 - torch.arange(n, device="cuda", dtype=torch.float32) * 0.0005
expect = torch.addcmul(y, x, torch.full_like(x, 2.5))      # 2.5*x + y, rounded once per element on CUDA
saxpy.Kernel().launch(n, 2.5, saxpy.from_torch(x, n), saxpy.from_torch(y, n))
torch.cuda.synchronize()
worst = float((y - expect).abs().max())
print(f"Python: max |y - (2.5x + y)| = {worst:g}, contract {saxpy.DERIVED_INTENSITY:.4f} flop/byte")
raise SystemExit(0 if worst <= 1e-6 else 1)
