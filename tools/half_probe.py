#!/usr/bin/env python3
"""Does this machine's `cvt.rn.f16.f32` agree with the oracle's rounding? ADR-0024 step 2.

    python tools/half_probe.py                 # check, and write fixtures/half_vectors.json

The rule this exists for is ADR-0024's third decision, and it is the project's oldest one
wearing a new hat: **the oracle models what the device will do, not what it should.** An
oracle that rounds a hair differently from the hardware produces a mismatch that is nobody's
bug, in a benchmark that then measures the disagreement instead of the kernel.

So before `lyth` emits a single `cvt`, the conversion is checked three ways against each other:

* **the device**, through a kernel that does nothing but `cvt.rn.f16.f32` and
  `cvt.rn.bf16.f32` in inline PTX -- the exact instructions the emitter will write, rather than
  an intrinsic that ought to lower to them -- and hands back the raw bits;
* **numpy**, whose `float16` is an independent IEEE implementation;
* **the oracle in `crates/lyth-lang/src/half.rs`**, via the fixture this writes.

All three must agree **bit for bit**, not approximately. The vectors are chosen to be the cases
where implementations diverge rather than a random sample, because a random sample of floats
almost never lands on a tie and would pass against a truncating implementation.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import struct
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

from nvrtc import compile_ptx  # noqa: E402
from vs_handwritten import Module, driver  # noqa: E402

# Inline PTX rather than `__float2half_rn` from `cuda_fp16.h`, for two reasons and only
# incidentally because NVRTC has no include path here. ADR-0024 says the emitter will write
# `cvt.rn.f16.f32`, so **that is the instruction this probe must measure** -- an intrinsic that
# should lower to it is one more "should" between the oracle and the silicon. And it keeps the
# probe readable as the thing it is checking.
SOURCE = """
extern "C" __global__ void convert(unsigned int n, const float* x,
                                   unsigned short* h, unsigned short* b)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    unsigned short a, c;
    asm("cvt.rn.f16.f32 %0, %1;"  : "=h"(a) : "f"(v));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(c) : "f"(v));
    h[i] = a;
    b[i] = c;
}
"""


def vectors() -> list[float]:
    """The cases that separate implementations, not a random sample.

    A uniform sample of floats essentially never lands on a tie, so it passes against a
    truncating conversion and proves nothing. Every entry here is chosen because some plausible
    implementation gets it wrong.
    """
    v: list[float] = []

    # Exactly representable, including the largest finite binary16.
    v += [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, -0.25, 1024.0, 65504.0, -65504.0]

    # Ties. Between 2048 and 4096 binary16 steps by 2, so every odd integer there is a tie and
    # round-half-to-even sends alternate ones in opposite directions.
    v += [2049.0, 2051.0, 2053.0, 2055.0, 2050.0, 2052.0]

    # Overflow. 65520 is the midpoint to a value that does not exist, so it becomes infinity
    # rather than saturating at 65504 -- a difference a reduction will notice.
    v += [65504.5, 65519.9, 65520.0, 65521.0, 70000.0, -70000.0, 3.0e38, float("inf")]

    # Subnormals. 2^-24 is the smallest binary16 subnormal; either side of half of it decides
    # whether the implementation flushes to zero.
    t = 2.0 ** -24
    v += [t, t * 0.4, t * 0.5, t * 0.6, t * 1.5, t * 2.5, 2.0 ** -14, 2.0 ** -15]

    # The normal/subnormal boundary itself, from both sides.
    v += [6.09e-5, 6.10e-5, 6.104e-5]

    # NaN, and a NaN whose low bits would carry into the exponent under a careless round.
    v += [float("nan"), struct.unpack("<f", struct.pack("<I", 0x7F80_0001))[0]]

    # bf16 cares about a different region: it has f32's range and 7 mantissa bits, so its ties
    # are at the 16-bit boundary of the significand.
    for bits in (0x3F80_8000, 0x3F81_8000, 0x3F80_7FFF, 0x3F80_8001, 0x4000_8000):
        v.append(struct.unpack("<f", struct.pack("<I", bits))[0])

    # And a deterministic spread, so the fixture is not only edge cases.
    rng = np.random.default_rng(24)
    v += list(rng.normal(0, 40, 64).astype(np.float32))
    v += list((rng.normal(0, 1, 32) * 1e-6).astype(np.float32))
    return v


def main() -> int:
    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    xs = np.array(vectors(), dtype=np.float32)
    n = len(xs)

    x = torch.from_numpy(xs).cuda()
    h = torch.zeros(n, dtype=torch.int16, device="cuda")
    b = torch.zeros(n, dtype=torch.int16, device="cuda")
    torch.cuda.synchronize()

    cuda = driver()
    mod = Module(cuda, compile_ptx(SOURCE, "half_probe.cu"))
    block = 128
    mod.launcher(
        "convert",
        [ctypes.c_uint32(n), ctypes.c_uint64(x.data_ptr()),
         ctypes.c_uint64(h.data_ptr()), ctypes.c_uint64(b.data_ptr())],
        (n + block - 1) // block, block,
    )()
    torch.cuda.synchronize()

    dev_h = h.cpu().numpy().view(np.uint16)
    dev_b = b.cpu().numpy().view(np.uint16)
    np_h = xs.astype(np.float16).view(np.uint16)

    # numpy has no bfloat16, so the reference for that column is the device alone and the Rust
    # oracle is checked against it directly. Said rather than quietly skipped.
    bad = []
    for i, (v, d, m) in enumerate(zip(xs, dev_h, np_h)):
        if d != m and not (np.isnan(v) and (d & 0x7C00) == 0x7C00 and (m & 0x7C00) == 0x7C00):
            bad.append((float(v), int(d), int(m)))

    print(f"  {n} vectors, binary16")
    if bad:
        print(f"  *** device and numpy disagree on {len(bad)} of them ***")
        for v, d, m in bad[:8]:
            print(f"    {v!r:>16}  device 0x{d:04x}  numpy 0x{m:04x}")
        return 1
    print("  device == numpy, bit for bit, including every tie, subnormal and overflow")

    out = REPO / "fixtures" / "half_vectors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": "Written by tools/half_probe.py. Each entry is an f32 bit pattern and the "
                "binary16 and bfloat16 bits THIS DEVICE produced for it with round-to-nearest-"
                "even. crates/lyth-lang/tests/half_rounding.rs checks the oracle against these. "
                "Measured, not derived: ADR-0024 decision 3.",
        "device": torch.cuda.get_device_name(0),
        "vectors": [
            {"f32": int(np.float32(v).view(np.uint32)), "f16": int(dh), "bf16": int(db)}
            for v, dh, db in zip(xs, dev_h, dev_b)
        ],
    }, indent=1), encoding="utf-8")
    print(f"  wrote {out.relative_to(REPO)} -- {n} vectors for the oracle to match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
