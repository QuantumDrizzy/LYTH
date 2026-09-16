#!/usr/bin/env python3
"""LYTH against a hand-written CUDA C++ matmul of the same schedule. ADR-0021 step 5.

    python bench/vs_handwritten_matmul.py --rounds 7
    ncu --metrics smsp__thread_inst_executed.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \\
        -k regex:matmul --csv python bench/vs_handwritten_matmul.py --once

ADR-0020 measured LYTH emitting **17% more instructions than nvcc and losing nothing for it**,
because every kernel this language could write waits on memory, and closed by naming the debt:

    > It would decide a compute-bound kernel's time, and this language cannot write one. So 17%
    > is a debt recorded rather than a debt paid, and the ADR that raises intensity is where it
    > comes due.

This is the collection notice. `tile 64, 64` + `coarsen 2, 2` is 16 flop/byte -- twice anything
before it, four multiply-adds on four shared loads per term instead of one on two -- and it is
the first kernel here where instructions could plausibly decide the time.

Two comparisons, and they are not the same question:

* **LYTH vs nvrtc `--fmad=false`.** Same schedule, same rounding, **the same bits** -- the
  harness refuses to time anything until `torch.equal` says so. Any difference is code
  generation, which is what ADR-0020 was about.
* **nvrtc at its default `fmad`.** A different function: one rounding per term instead of two.
  That number is the price of ADR-0010's rule, measured, and it belongs to that decision rather
  than to this one. Reported separately and never folded into the first.

And both LYTH tiles, because the point of ADR-0021 is the step from 8 flop/byte to 16, and a
comparison at only the new one cannot show whether the gap moved.

**Correctness first.** Three kernels have reported impossible throughput in this project and
each one looked plausible: 3924 GB/s from a launcher whose arguments Python had already freed,
1646 GB/s from counting bytes twice, 9.2 GB/s from a grid of one. The launchers checked here are
the launchers timed.
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
from gpu_health import Watch  # noqa: E402
from vs_handwritten import Module, driver, timed  # noqa: E402

# (tile, per-axis coarsening, generated binding, hand-written entry point, label)
#
# `tile 16` is not decoration. Coarsening halves the global traffic and the shared-load count by
# the same factor of two, so the 32-vs-64 pair cannot say which of them paces the kernel.
# `tile 16` doubles the traffic per output and leaves the shared-load count per output
# *unchanged* -- one thread per tile element either way, two loads per term -- so the two
# hypotheses predict times a factor of two apart.
VARIANTS = [
    (16, (1, 1), "matmul16", "matmul_t16", "tile 16"),
    (32, (1, 1), "matmul32", "matmul_t32", "tile 32"),
    (64, (2, 2), "matmul64c2", "matmul_t64c2", "tile 64 + coarsen 2,2"),
    # ADR-0022 step 4: predicted at 3.92 and 5.76 TFLOP/s before either was run, at 512 and
    # 256 threads per block.
    (64, (2, 4), "matmul64c24", "matmul_t64c24", "tile 64 + coarsen 2,4"),
    (64, (4, 4), "matmul64c44", "matmul_t64c44", "tile 64 + coarsen 4,4"),
]


def build_binding(
    binary: pathlib.Path, machine: pathlib.Path, tile: int, c: tuple[int, int], mod: str
):
    """Generate the source for one tile, compile it, and import its binding.

    The source is `examples/matmul.lyth` with three declarations rewritten, exactly as
    `tools/contraction_traffic.py` does it -- the declared asymptote has to move with the tile
    (`T/4`) or the compiler refuses the file, which is the compiler doing its job.
    """
    gen = HERE / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    path = gen / f"{mod}.py"
    if not path.exists():
        if not binary.exists():
            sys.exit(f"{binary} is not built. Run `cargo build --release`.")
        src = (REPO / "examples/matmul.lyth").read_text(encoding="utf-8")
        nl = chr(10)
        tile_line = f"    tile {tile}, {tile}{nl}"
        if c != (1, 1):
            tile_line += f"    coarsen {c[0]}, {c[1]}{nl}"
        src = src.replace(f"    tile 32, 32{nl}", tile_line)
        src = src.replace(
            f"    intensity asymptotic 8.0{nl}",
            f"    intensity asymptotic {tile / 4}{nl}",
        )
        lyth_src = gen / f"{mod}.lyth"
        lyth_src.write_text(src, encoding="utf-8")
        subprocess.run(
            [str(binary), "build", str(lyth_src), "--machine", str(machine),
             "-o", "nul" if sys.platform == "win32" else "/dev/null",
             "--bind-py", str(path)],
            check=True, capture_output=True,
        )
    return importlib.import_module(mod)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", type=int, default=2048, help="m = n = k")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument(
        "--once", action="store_true",
        help="launch each kernel exactly once and exit -- the shape ncu wants",
    )
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--machine", type=pathlib.Path, default=REPO / "fixtures/machine/sm_120.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    sz = args.size
    torch.manual_seed(5)
    a = torch.randn(sz, sz, device="cuda", dtype=torch.float32)
    b = torch.randn(sz, sz, device="cuda", dtype=torch.float32)

    cuda = driver()
    # Two builds of the same source. `exact` is the comparison; `fused` is ADR-0010's price.
    cu = (HERE / "cuda" / "handwritten_matmul.cu").read_text(encoding="utf-8")
    hand = {
        "exact": Module(cuda, compile_ptx(cu, "hm.cu", extra=("--fmad=false",))),
        "fused": Module(cuda, compile_ptx(cu, "hm.cu")),
    }

    calls: dict[str, callable] = {}
    outs: dict[str, torch.Tensor] = {}

    for tile, c, mod, entry, label in VARIANTS:
        lyth = build_binding(args.binary, args.machine, tile, c, mod)
        grid, block, shared = lyth.grid(sz, sz, sz), lyth.BLOCK, lyth.SHARED_BYTES

        cl = torch.zeros(sz, sz, device="cuda", dtype=torch.float32)
        k = lyth.Kernel()
        calls[f"LYTH {label}"] = lambda k=k, cl=cl: k.launch(
            sz, sz, sz, a.data_ptr(), b.data_ptr(), cl.data_ptr()
        )
        outs[f"LYTH {label}"] = cl

        for build, m in hand.items():
            ch = torch.zeros(sz, sz, device="cuda", dtype=torch.float32)
            name = f"nvrtc {label}" + ("" if build == "exact" else " [fmad]")
            calls[name] = m.launcher(
                entry,
                [ctypes.c_uint32(sz), ctypes.c_uint32(sz), ctypes.c_uint32(sz),
                 ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(b.data_ptr()),
                 ctypes.c_uint64(ch.data_ptr())],
                grid, block, shared,
            )
            outs[name] = ch

        print(f"  {label:<22} grid {grid:>5} x block {block}, {shared:,} B shared "
              f"(LYTH's launch, used for both)")

    if args.once:
        # One launch each, nothing else. Under ncu this is the whole run.
        for fn in calls.values():
            fn()
        torch.cuda.synchronize()
        return 0

    # --- correctness, before anything is timed ------------------------------------------
    print()
    ref = (a.double() @ b.double()).float()
    for name, fn in calls.items():
        fn()
    torch.cuda.synchronize()
    for name in calls:
        rel = ((outs[name] - ref).abs().max() / ref.abs().max()).item()
        print(f"  {name:<32} max rel err vs float64 {rel:.2e}")
        if rel > 1e-4:
            sys.exit(f"{name} computes the wrong thing; a timing would mean nothing")

    # The claim that makes the first comparison a comparison: same schedule, same rounding
    # rule, identical bits. If this fails, one of the two is not following the other's
    # schedule and the timing below would be about two different kernels.
    print()
    for *_, label in VARIANTS:
        pair = (f"LYTH {label}", f"nvrtc {label}")
        same = torch.equal(outs[pair[0]], outs[pair[1]])
        print(f"  {label:<22} LYTH == nvrtc --fmad=false : "
              f"{'BIT-IDENTICAL' if same else '*** DIFFER ***'}")
        if not same:
            sys.exit("the two kernels disagree; they are not running the same schedule")
        fused = outs[f"nvrtc {label} [fmad]"]
        n_diff = (fused != outs[pair[0]]).sum().item()
        print(f"  {'':<22} and the fused build differs in {n_diff:,} of {sz * sz:,} outputs "
              f"-- a different function, as ADR-0010 says")

    # --- interleaved, same grid, same block, order rotated each round --------------------
    names = list(calls)
    rows = {k: [] for k in names}
    flops = 2.0 * sz * sz * sz
    # ADR-0023. Sixteen display-driver resets happened during this project's benchmarking on
    # one day and not one of them reached a benchmark's output. A run that spans a reset keeps
    # timing and keeps printing.
    watch = Watch()
    watch.__enter__()
    for r in range(args.rounds):
        # Rotating the order matters: a fixed one biased the last kernel by 56% in ADR-0019's
        # bandwidth run, on traffic that was identical by construction.
        for i in range(len(names)):
            label = names[(r + i) % len(names)]
            rows[label].append(flops / timed(calls[label], args.reps) / 1e12)
        print(f"  round {r + 1}/{args.rounds}", file=sys.stderr)

    print(f"\n  m = n = k = {sz}, {flops / 1e9:.1f} GFLOP per launch, "
          f"{args.rounds} rounds x {args.reps} launches, interleaved\n")
    print(f"  {'':<32} {'TFLOP/s median':>15} {'min':>8} {'max':>8}")
    med = {}
    for name in names:
        v = sorted(rows[name])
        med[name] = st.median(v)
        print(f"  {name:<32} {med[name]:>15.2f} {v[0]:>8.2f} {v[-1]:>8.2f}")

    print()
    for *_, label in VARIANTS:
        l, h = f"LYTH {label}", f"nvrtc {label}"
        lo = min(rows[l]) / max(rows[h])
        hi = max(rows[l]) / min(rows[h])
        print(f"  {label:<22} LYTH / nvrtc = {med[l] / med[h]:.1%}  [{lo:.0%}-{hi:.0%}]")
        f = f"nvrtc {label} [fmad]"
        print(f"  {'':<22} and against the fused build, {med[l] / med[f]:.1%} "
              f"-- ADR-0010's price, not a code-generation result")
    print()
    watch.report()
    # A run that crossed a driver reset is not a failed run to be retried quietly: it is a run
    # whose numbers are not about what they claim, and the exit code says so.
    return 0 if watch.clean is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
