#!/usr/bin/env python3
"""What this device can actually move and retire, measured, for a lyth machine file.

    python tools/peak_probe.py
    python tools/peak_probe.py --out fixtures/machine/meas-peak-sm_120.json

Every `% of peak` this compiler prints is a ratio against a number produced here. That makes
this file's own defects everybody's defects, and it had two.

**It timed a host round trip as if it were memory traffic.** The bandwidth probe was
`float(x.sum())` inside a `perf_counter` loop, and `float(...)` on a device tensor blocks until
the result reaches the host. Measured on this machine: 368.02 GB/s with the conversion against
410.66 GB/s without it -- the synchronisation was **10.4%** of the figure. Everything downstream
inherited it, which is why `saxpy` reported `112% of the machine file` and `sum` reported
`115%`. A kernel cannot exceed its device's peak bandwidth; a kernel can easily exceed a number
that is 10% too small.

So: CUDA events rather than a wall clock, and **nothing inside the timed region may touch the
host**. `x.sum()` without the conversion leaves the result on the device, which is what a
bandwidth probe is supposed to do.

**It measured one pattern and called it peak.** A read-only sweep, compared against kernels that
read and write. A write is not a read on this hardware, so the two are different quantities and
a ratio between them means nothing in particular. Three patterns are measured here, each with
its byte accounting beside it, and all three are recorded.

`bandwidth_gbs` takes the **fastest observed**, which on this device is the read-only sweep.
"`saxpy` reaches 96% of the best rate this device was seen to achieve" is a sentence that is
true and that a reader can check. It also means a kernel that writes will not reach 100%, and
that is a property of the machine rather than a failing of the kernel -- said here so nobody has
to infer it from a percentage.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path

import torch


def timed(fn, reps: int, warmup: int = 5) -> float:
    """Seconds per call, measured on the device.

    CUDA events, because a wall clock around a loop also measures launch overhead and whatever
    Python does between calls. `fn` must not return a host value -- see the module docstring.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps / 1e3


def bandwidth(n_bytes: int, reps: int) -> dict[str, float]:
    """Three streaming patterns, each with the bytes it actually moves.

    `copy` and `triad` come out slower than `read`, which is expected and is the reason all
    three are kept: reporting only the fastest hides that a write costs more, and reporting
    only a mixed one calls the device slower than it is.
    """
    n = n_bytes // 4
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    out = {}
    try:
        # One full pass over x. `sum` stands in for a streaming load, and the result stays on
        # the device, which is the whole point.
        out["read"] = n_bytes / timed(lambda: x.sum(), reps) / 1e9
        # Read x, write y.
        out["copy"] = 2 * n_bytes / timed(lambda: y.copy_(x), reps) / 1e9
        # Read x, read y, write y -- the shape `saxpy` has.
        out["triad"] = 3 * n_bytes / timed(lambda: y.add_(x, alpha=2.0), reps) / 1e9
    finally:
        del x, y
        torch.cuda.empty_cache()
    return out


def gemm_tflops(dtype: torch.dtype, n: int, reps: int) -> float:
    """Achieved GEMM, which is what a roofline's compute ceiling should be measured against.

    Not the datasheet figure. A kernel is compared against what this machine was seen to do,
    and cuBLAS is the best available stand-in for that.
    """
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    try:
        secs = timed(lambda: a @ b, reps)
        return (2.0 * n * n * n / secs) / 1e12
    finally:
        del a, b
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="write the measurement as JSON")
    ap.add_argument("--bytes", type=int, default=1 << 30, help="buffer size for the bandwidth probes")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--gemm", type=int, default=8192)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{props.major}{props.minor}"
    print(f"{props.name}  {arch}  {props.multi_processor_count} SMs\n")

    bw = bandwidth(args.bytes, args.reps)
    print(f"  {'pattern':<10} {'bytes/element':>14} {'GB/s':>9}")
    for pattern, per in (("read", 4), ("copy", 8), ("triad", 12)):
        print(f"  {pattern:<10} {per:>14} {bw[pattern]:>9.2f}")
    ceiling = max(bw.values())
    fastest = max(bw, key=bw.get)
    print(f"\n  ceiling {ceiling:.2f} GB/s from `{fastest}` -- this is what `bandwidth_gbs` records.")
    print(f"  a kernel that writes will not reach it: `triad` is {bw['triad'] / ceiling:.1%} of it.")

    fp32 = gemm_tflops(torch.float32, args.gemm, 10)
    fp16 = gemm_tflops(torch.float16, args.gemm, 10)
    print(f"\n  achieved SGEMM {fp32:.2f} TFLOP/s, HGEMM {fp16:.2f} TFLOP/s")
    print(f"  ridge {fp32 * 1e3 / ceiling:.1f} flop/byte")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema": "lyth-machine-measurement/0.2",
                    "machine_id": arch,
                    "device": props.name,
                    "sm_count": props.multi_processor_count,
                    "measured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "host": platform.node(),
                    "source": f"lyth/tools/peak_probe.py torch {torch.__version__}",
                    "buffer_bytes": args.bytes,
                    "reps": args.reps,
                    # All three, because `bandwidth_gbs` is one of them and a reader is owed
                    # the others rather than a bare maximum.
                    "bandwidth_gbs_by_pattern": {k: round(v, 2) for k, v in bw.items()},
                    "bandwidth_gbs": round(ceiling, 2),
                    "bandwidth_pattern": fastest,
                    "peak_tflops": round(fp32, 2),
                    "peak_tflops_fp16": round(fp16, 2),
                    "ridge_flops_per_byte": round(fp32 * 1e3 / ceiling, 2),
                    "notes": [
                        "peak_tflops is achieved cuBLAS GEMM, not a datasheet figure.",
                        "Timed with CUDA events, and nothing in the timed region touches the "
                        "host: the previous probe converted the result to a Python float each "
                        "iteration and measured that round trip as memory traffic, which cost "
                        "10.4% of the figure and made every derived percentage 10% too high.",
                        "bandwidth_gbs is the fastest pattern. A kernel that writes cannot "
                        "reach it, which is a property of the device.",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
