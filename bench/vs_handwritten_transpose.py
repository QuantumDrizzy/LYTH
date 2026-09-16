#!/usr/bin/env python3
"""LYTH's tiled transpose against a hand-written one, and both against the naive form.

    python bench/vs_handwritten_transpose.py --rounds 7

`vs_handwritten.py` compared a `saxpy`, which is the easiest kernel either compiler will ever
emit: a loop, a load, an fma, a store. This is the hard one. A tiled transpose has shared
memory, two barriers, a skewed row stride, boundary guards on two axes and a permutation that
has to happen in the right place. It is where a generated kernel is most likely to be worse
than one a person wrote, and it is the kernel LYTH's central claim is about --
**`tile 32, 32` is two declared lines that turn 36 bytes per element into 8.**

Four kernels, one process, interleaved:

    LYTH  tiled      examples/transpose-tiled.lyth, through its generated binding
    nvcc  tiled      bench/cuda/handwritten.cu, the textbook 32x32 with a +1 skew
    LYTH  naive      examples/transpose.lyth -- the strided store, no tile
    nvcc  naive      the same schedule by hand

The naive pair is not filler. Without it, "the tiled one is faster" is a claim about tiling that
neither compiler is responsible for; with it, the same ratio appears on both sides and what is
left is the difference between the compilers.

**Each kernel is checked against `torch.t().contiguous()` before it is timed**, and the
launchers that are checked are the launchers that are timed -- building a second one for the
check is how an argument-lifetime bug hid in the saxpy version of this file and reported
3924 GB/s.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import pathlib
import statistics as st
import subprocess
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "gen"))
sys.path.insert(0, str(HERE))

from nvrtc import compile_ptx  # noqa: E402
from vs_handwritten import Module, driver, timed  # noqa: E402

BINDINGS = {
    "transpose_tiled": "transpose-tiled.lyth",
    "transpose_flat": "transpose.lyth",
}


def ensure(binary: pathlib.Path, machine: pathlib.Path) -> None:
    gen = HERE / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    for mod, src in BINDINGS.items():
        if (gen / f"{mod}.py").exists():
            continue
        if not binary.exists():
            sys.exit(f"{binary} is not built. Run `cargo build --release`.")
        subprocess.run(
            [str(binary), "build", str(REPO / "examples" / src), "--machine", str(machine),
             "-o", "nul" if sys.platform == "win32" else "/dev/null",
             "--bind-py", str(gen / f"{mod}.py")],
            check=True, capture_output=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=2048, help="rows = cols")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--machine", type=pathlib.Path, default=REPO / "fixtures/machine/sm_120.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    ensure(args.binary, args.machine)
    tiled = importlib.import_module("transpose_tiled")
    flat = importlib.import_module("transpose_flat")

    n = args.size
    torch.manual_seed(5)
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    want = a.t().contiguous()

    cuda = driver()
    hand = Module(cuda, compile_ptx((HERE / "cuda" / "handwritten.cu").read_text(encoding="utf-8"), "h.cu"))

    outs = {k: torch.empty_like(a) for k in ("lyth_tiled", "nvcc_tiled", "lyth_naive", "nvcc_naive")}
    kt, kf = tiled.Kernel(), flat.Kernel()

    def u32(v):
        return ctypes.c_uint32(v)

    def u64(v):
        return ctypes.c_uint64(v)

    calls = {
        # LYTH's own launch shape, read from the binding rather than chosen here.
        "lyth_tiled": (
            lambda: kt.launch(n, n, a.data_ptr(), outs["lyth_tiled"].data_ptr()),
            f"grid {tiled.grid(n, n):,} x {tiled.BLOCK}, {tiled.SHARED_BYTES} B shared",
        ),
        "nvcc_tiled": (
            hand.launcher("transpose_tiled",
                          [u32(n), u32(n), u64(a.data_ptr()), u64(outs["nvcc_tiled"].data_ptr())],
                          tiled.grid(n, n), tiled.BLOCK),
            f"grid {tiled.grid(n, n):,} x {tiled.BLOCK}, static __shared__",
        ),
        "lyth_naive": (
            lambda: kf.launch(n, n, a.data_ptr(), outs["lyth_naive"].data_ptr()),
            f"grid {flat.grid(n, n):,} x {flat.BLOCK}",
        ),
        "nvcc_naive": (
            hand.launcher("transpose_naive",
                          [u32(n), u32(n), u64(a.data_ptr()), u64(outs["nvcc_naive"].data_ptr())],
                          flat.grid(n, n), flat.BLOCK),
            f"grid {flat.grid(n, n):,} x {flat.BLOCK}",
        ),
    }

    print(f"\n  {n} x {n} f32, {n * n * 4 / 1e6:.0f} MB per buffer\n")
    for label, (fn, shape) in calls.items():
        fn()
        torch.cuda.synchronize()
        ok = torch.equal(outs[label], want)
        print(f"  {label:<12} {'transposes correctly' if ok else '*** WRONG ***':<22} {shape}")
        if not ok:
            sys.exit(f"{label} is wrong; a timing would mean nothing")
    print()

    # 8 bytes per element is the payload both forms move: 4 read, 4 written. The naive form
    # moves far more than that across the bus, which is the whole point -- so this figure is
    # the *payload* rate, not a bus rate, and the two are only equal when nothing is wasted.
    elems = n * n
    rows = {k: [] for k in calls}
    order = list(calls)
    for r in range(args.rounds):
        # **Rotate the order every round.** Interleaving equalises the clocks, and with four
        # kernels writing 268 MB each it introduces a second unfairness in its place: whichever
        # runs last always inherits the worst cache and DRAM state. In a fixed order that is
        # the same kernel every round.
        #
        # Measured: at 8192 a fixed order made the last kernel look 56% slower than one that
        # moves byte-for-byte identical traffic at L2 *and* at DRAM. Run alone, the two are
        # 100.2% apart. The gap was the harness.
        turn = order[r % len(order):] + order[: r % len(order)]
        for label in turn:
            rows[label].append(8 * elems / timed(calls[label][0], args.reps) / 1e9)
        print(f"  round {r + 1}/{args.rounds}", file=sys.stderr)

    print(f"  {'':<12} {'payload GB/s':>13} {'min':>8} {'max':>8}")
    med = {}
    for label in calls:
        v = sorted(rows[label])
        med[label] = st.median(v)
        print(f"  {label:<12} {med[label]:>13.1f} {v[0]:>8.1f} {v[-1]:>8.1f}")

    def band(x, y):
        lo = min(rows[x]) / max(rows[y])
        hi = max(rows[x]) / min(rows[y])
        return f"{med[x] / med[y]:.1%}  [{lo:.0%}-{hi:.0%}]"

    print(f"\n  tiled:  LYTH / nvcc = {band('lyth_tiled', 'nvcc_tiled')}")
    print(f"  naive:  LYTH / nvcc = {band('lyth_naive', 'nvcc_naive')}")
    print(f"\n  what the tile buys, on each side:")
    print(f"    LYTH  tiled / naive = {band('lyth_tiled', 'lyth_naive')}")
    print(f"    nvcc  tiled / naive = {band('nvcc_tiled', 'nvcc_naive')}")
    print(f"\n  {args.rounds} rounds x {args.reps} launches, interleaved, one process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
