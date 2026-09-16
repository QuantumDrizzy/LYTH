#!/usr/bin/env python3
"""The shared-memory access rate of this device, from a kernel that does nothing else.

    python tools/shared_probe.py

ADR-0022 step 1. ADR-0021 measured six matmul kernels pacing at ~40 G shared accesses per
second while their L2 traffic varied by 118% and their instruction counts by 49%, and proposed
that the shared pipe is what binds a tiled contraction. **That rate cannot be taken from those
runs.** It was read off them, so using it to predict them is a definition dressed as a
prediction. This measures it independently.

**The unit is accesses, not bytes, and the kernel is built to prove it.** One kernel, one
instruction stream, a runtime `mask` that is the only difference between the two patterns:

* `mask = 31` -- each lane reads a different address, 32 consecutive floats, 128 useful bytes
  per wavefront.
* `mask = 0` -- every lane reads the *same* address, a broadcast, **4** useful bytes per
  wavefront.

Same instructions, same access count, 32x different payload. If the two rates agree, the pipe
is priced in wavefronts and a `bandwidth_gbs` for this level would be the wrong division --
which is what ADR-0022 claims and `fixtures/machine/sm_120.json` has left at `0.0` all along.
If broadcast is faster, the claim is wrong and the ADR says so instead.

**Two controls, because a probe that measures nothing looks exactly like a fast one:**

1. *Scaling.* Every point is run at `iters` and `2 * iters`. If the loads were hoisted out of
   the loop, doubling the work would not double the time. Anything outside 1.8-2.2x is rejected
   rather than reported.
2. *The PTX is counted.* Eight `ld.shared.f32` per iteration is asserted on the emitted code,
   not assumed from the source.

`tools/peak_probe.py` learned the first lesson the hard way: `float(x.sum())` inside the timed
region cost 10.4%, and every "% of peak" in the project was that much too high until it came
out.
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import statistics as st
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

from nvrtc import compile_ptx  # noqa: E402
from gpu_health import Watch  # noqa: E402
from vs_handwritten import Module, driver, timed  # noqa: E402

LOADS_PER_ITER = 8

# Addresses depend on `i`, so nothing is loop-invariant and nothing can be hoisted; they depend
# on `lane & mask`, so the same instruction stream serves both patterns. `& 1023` keeps every
# access inside the tile without a branch. `#pragma unroll 1` holds the loop shape fixed so the
# two patterns cannot be unrolled differently.
SOURCE = """
extern "C" __global__ void shared_probe(unsigned int iters, unsigned int mask, float* out)
{
    extern __shared__ float tile[];

    for (unsigned int i = threadIdx.x; i < 1024u; i += blockDim.x) tile[i] = (float)(i + 1u);
    __syncthreads();

    const unsigned int p = threadIdx.x & mask;
    float acc = 0.0f;

#pragma unroll 1
    for (unsigned int i = 0; i < iters; ++i) {
        const unsigned int b = (p + i * 32u) & 1023u;
        acc += tile[b];
        acc += tile[(b + 128u) & 1023u];
        acc += tile[(b + 256u) & 1023u];
        acc += tile[(b + 384u) & 1023u];
        acc += tile[(b + 512u) & 1023u];
        acc += tile[(b + 640u) & 1023u];
        acc += tile[(b + 768u) & 1023u];
        acc += tile[(b + 896u) & 1023u];
    }

    // Never taken -- `acc` is positive by construction -- but the compiler cannot know that, so
    // every load stays. Writing unconditionally would add a global store to the timed loop.
    if (acc < 0.0f) out[blockIdx.x] = acc;
}
"""


def count_shared_loads(ptx: str) -> int:
    inside = False
    n = 0
    for line in ptx.splitlines():
        t = line.strip()
        if ".entry" in t:
            inside = "shared_probe" in t
            continue
        if inside:
            if t == "}":
                break
            n += t.count("ld.shared.f32")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iters", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks-per-sm", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    ptx = compile_ptx(SOURCE, "shared_probe.cu")
    got = count_shared_loads(ptx)
    if got != LOADS_PER_ITER:
        sys.exit(
            f"control failed: the PTX holds {got} `ld.shared.f32`, not {LOADS_PER_ITER}. "
            "Either the loop was unrolled or the loads were folded, and the rate below would "
            "be about the wrong number of accesses."
        )
    print(f"  control  {got} ld.shared.f32 per iteration, in the emitted PTX")

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    # Allocate first. torch creates its CUDA context lazily, and loading a module before that
    # happens fails with `invalid device context` -- the module is loaded into the current
    # context, and until torch touches the device there is not one.
    out = torch.zeros(4096, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    cuda = driver()
    mod = Module(cuda, ptx)

    print(f"\n  {sms} SMs, {args.blocks_per_sm} blocks each, {args.iters:,} iterations "
          f"x {LOADS_PER_ITER} loads\n")
    print(f"  {'block':>6} {'pattern':<11} {'Gaccess/s':>11} {'min':>8} {'max':>8} {'scaling':>9}")

    rates: dict[tuple[int, str], float] = {}
    # ADR-0023: a probe that ran across a display-driver reset measured something else.
    watch = Watch().start()
    for block in (256, 1024):
        grid = sms * args.blocks_per_sm
        for pattern, mask in (("coalesced", 31), ("broadcast", 0)):
            runs = []
            scale = None
            for iters in (args.iters, args.iters * 2):
                launch = mod.launcher(
                    "shared_probe",
                    [ctypes.c_uint32(iters), ctypes.c_uint32(mask),
                     ctypes.c_uint64(out.data_ptr())],
                    grid, block, 1024 * 4,
                )
                accesses = grid * block * LOADS_PER_ITER * iters
                v = [accesses / timed(launch, args.reps) / 1e9 for _ in range(args.rounds)]
                if iters == args.iters:
                    runs = sorted(v)
                else:
                    # Control 1: twice the iterations, twice the accesses. If the rate is the
                    # same, the time doubled and the loop really ran.
                    scale = st.median(runs) / st.median(sorted(v))
            assert scale is not None
            med = st.median(runs)
            rates[(block, pattern)] = med
            flag = "" if 0.90 <= scale <= 1.10 else "  *** NOT SCALING ***"
            print(f"  {block:>6} {pattern:<11} {med:>11.1f} {runs[0]:>8.1f} {runs[-1]:>8.1f} "
                  f"{scale:>8.2f}x{flag}")
            if not 0.90 <= scale <= 1.10:
                sys.exit(
                    "control failed: doubling the iterations did not halve the throughput, so "
                    "the timed region is not the loop."
                )

    print()
    for block in (256, 1024):
        c, b = rates[(block, "coalesced")], rates[(block, "broadcast")]
        print(f"  block {block:>4}: broadcast / coalesced = {b / c:.1%} "
              f"-- payload differs 32x, rate differs {abs(b / c - 1) * 100:.0f}%")

    # The median across every configuration, not the best of them. The four agree to well
    # under a percent, which is the finding: the rate is a property of the pipe and not of the
    # block shape or the access pattern, so picking the largest would be quoting noise.
    v = sorted(rates.values())
    med = st.median(v)
    spread = (v[-1] / v[0] - 1) * 100
    print(f"\n  {med:.1f} G thread-accesses/s = {med / 32:.1f} G wavefronts/s")
    print(f"  spread across all four configurations: {spread:.1f}%")
    print(f"  payload at that rate: {med * 4 / 1e3:.1f} TB/s coalesced, "
          f"{med / 32 * 4 / 1e3:.2f} TB/s broadcast -- a 32x difference at the same rate")
    print(f"\n  ADR-0021's matmuls paced at 39.4-43.4 G wavefronts/s, "
          f"{39.4 / (med / 32):.0%}-{43.4 / (med / 32):.0%} of this.")
    print()
    watch.report()
    return 0 if watch.clean is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
