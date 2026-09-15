#!/usr/bin/env python3
"""LYTH against its reference, measured back to back in one process.

    python bench/ab_bandwidth.py --rounds 7

**Why this exists.** `lyth run --time N` times a kernel well, and `tools/peak_probe.py` times a
reference well, and comparing the two was still wrong -- they run in different processes,
minutes or hours apart, with the card in a different thermal and clock state. Measured three
times on the same machine on one afternoon, the read-only reference came out at **414.51,
425.70 and 403.94 GB/s**: a 5.4% spread between runs against under 1% within a run. Every
"% of peak" figure derived across that gap carried 5% of unstated uncertainty and was printed
to one decimal.

So: **one process, interleaved, repeated**. Each round measures the LYTH kernel and the torch
reference of the *same traffic shape* one after the other, so both see the same clocks. The
ratio of a pair is then a ratio of two numbers taken seconds apart, and the spread over rounds
is reported rather than hidden behind a median.

LYTH runs here through its own generated Python bindings -- the same files a caller would use,
embedding the same PTX -- so this measures the kernel a user gets, not a private build.

**Reference shapes.** A kernel is paired with the reference that moves bytes the way it does:

    4 B/element, read only ...... torch.sum(x)          one stream in, no per-element store
    8 B/element, read only ...... torch.dot(x, y)       two streams in, no per-element store
    8 B/element, read+write ..... y.copy_(x)            one in, one out
    12 B/element, 2r+1w ......... y.add_(x, alpha=a)    two in, one out

`split` reads one buffer and writes two. Nothing in torch's elementwise set has that shape, so
it is reported against `triad` with the mismatch named rather than silently compared.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import statistics as st
import subprocess
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN = REPO / "bench" / "gen"
sys.path.insert(0, str(GEN))


def ensure_bindings(names, binary: pathlib.Path, machine: pathlib.Path) -> None:
    """Generate the Python bindings this file imports, if they are not already there.

    `bench/gen/` is build output and is not committed, so a fresh clone has nothing to import.
    Verified by cloning: `ModuleNotFoundError: No module named 'sum'`. A benchmark that only
    runs on the machine that wrote it is a benchmark nobody can check, which is the same class
    of defect as the CRLF one -- HEAD could not parse its own examples on a fresh clone either.
    """
    GEN.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (GEN / f"{n}.py").exists()]
    if not missing:
        return
    if not binary.exists():
        sys.exit(
            f"{binary} is not built. Run `cargo build --release` first, or pass --binary."
        )
    print(f"  generating {len(missing)} binding(s) into {GEN.relative_to(REPO)}", file=sys.stderr)
    for name in missing:
        src = REPO / "examples" / f"{name}.lyth"
        done = subprocess.run(
            [str(binary), "build", str(src), "--machine", str(machine),
             "-o", "nul" if sys.platform == "win32" else "/dev/null",
             "--bind-py", str(GEN / f"{name}.py")],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            sys.exit(f"{src.name} did not compile:" + done.stdout + done.stderr)

# kernel module, its manifest name, bytes/element, reference key, and how to call it
KERNELS = [
    ("sum",    4,  "read1", "reduce"),
    ("min",    4,  "read1", "reduce"),
    ("max",    4,  "read1", "reduce"),
    ("dot",    8,  "read2", "reduce2"),
    ("horner", 8,  "copy",  "map1"),
    ("lerp",   12, "triad", "map2"),
    ("axpby",  12, "triad", "map2"),
    ("saxpy",  12, "triad", "map2"),
    ("split",  12, "triad*", "split"),
]


def timed(fn, reps: int) -> float:
    """Seconds per call, on the device. Nothing inside touches the host."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1 << 26, help="elements")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--machine", type=pathlib.Path, default=REPO / "fixtures/machine/sm_120.json")
    args = ap.parse_args()

    ensure_bindings([name for name, *_ in KERNELS], args.binary, args.machine)

    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    n = args.n
    torch.manual_seed(11)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    ybak = y.clone()

    # The module is named after the source file; the PTX entry point inside it is
    # `max_of`/`min_of`, which is the binding's business and not this file's.
    mods = {name: importlib.import_module(name) for name, *_ in KERNELS}
    kernels = {name: m.Kernel() for name, m in mods.items()}

    # The bytes-per-element in KERNELS is what the labels and the ratios are built on. The
    # binding carries the compiler's own figure, so it is checked rather than trusted: a table
    # that drifted from the manifest would mislabel every bar without failing anything.
    for name, bpe, *_ in KERNELS:
        got = mods[name].BYTES_PER_ELEMENT
        if got != bpe:
            sys.exit(f"{name}: this file says {bpe} B/element, the binding says {got}")
    # Only a reduction has a target sized by the grid; the binding says so by having
    # `partial_len` at all, which is read rather than assumed.
    partials = {name: torch.empty(m.partial_len(n), device="cuda", dtype=torch.float32)
                for name, m in mods.items() if hasattr(m, "partial_len")}
    w = torch.empty_like(x)  # `split` writes two buffers

    def lyth_call(name):
        # Signatures come from the generated bindings, not from memory: `split` takes a scalar
        # and writes two buffers, `horner` takes four coefficients. Guessing one of these wrong
        # would still run and would time a different kernel than the label claims.
        k = kernels[name]
        px, py, pz, pw = x.data_ptr(), y.data_ptr(), z.data_ptr(), w.data_ptr()
        if name in ("sum", "min", "max"):
            pp = partials[name].data_ptr()
            return lambda: k.launch(n, px, pp)
        if name == "dot":
            pp = partials[name].data_ptr()
            return lambda: k.launch(n, px, py, pp)
        if name == "horner":
            return lambda: k.launch(n, 1.0, 2.0, 3.0, 4.0, px, pz)
        if name == "split":
            return lambda: k.launch(n, 0.5, px, pz, pw)
        if name == "saxpy":
            return lambda: k.launch(n, 2.0, px, py)
        if name == "axpby":
            return lambda: k.launch(n, 2.0, 3.0, px, py)
        if name == "lerp":
            return lambda: k.launch(n, 0.25, px, py, pz)
        raise KeyError(name)

    # `stream4` is the bytes one f32 stream moves over the whole buffer. A reference is a
    # count of streams; a kernel carries its own bytes per element. Multiplying a kernel's
    # bytes-per-element by `stream4` instead of by `n` inflates it exactly fourfold, which is
    # how the first run of this file reported 1646 GB/s on a 448 GB/s card -- the reference
    # column was right and only the kernel column was wrong, so the ratio came out a uniform
    # 4x across all nine rows, which is what gave it away.
    stream4 = n * 4
    refs = {
        "read1":  (lambda: x.sum(),                  1),
        "read2":  (lambda: torch.dot(x, y),          2),
        "copy":   (lambda: z.copy_(x),               2),
        "triad":  (lambda: y.add_(x, alpha=2.0),     3),
    }
    refs["triad*"] = refs["triad"]

    rows = {name: {"lyth": [], "ref": []} for name, *_ in KERNELS}
    for r in range(args.rounds):
        for name, bpe, refkey, _ in KERNELS:
            call = lyth_call(name)
            # Back to back, kernel then reference, so both see the same clocks.
            t_l = timed(call, args.reps)
            t_r = timed(refs[refkey][0], args.reps)
            rows[name]["lyth"].append(bpe * n / t_l / 1e9)
            rows[name]["ref"].append(refs[refkey][1] * stream4 / t_r / 1e9)
            y.copy_(ybak)  # `saxpy`/`axpby`/`triad` write into y; keep every round identical
        print(f"  round {r + 1}/{args.rounds} done", file=sys.stderr)

    def band(v):
        return st.median(v), min(v), max(v)

    print(f"\n  n = {n:,}   {args.rounds} rounds x {args.reps} launches, interleaved in one process\n")
    print(f"  {'kernel':<8} {'B/el':>4} {'LYTH GB/s':>18} {'reference':>22} {'ratio':>16}")
    out = []
    for name, bpe, refkey, _ in KERNELS:
        lm, llo, lhi = band(rows[name]["lyth"])
        rm, rlo, rhi = band(rows[name]["ref"])
        # The ratio of medians, with the band it could occupy given both spreads.
        lo, hi = llo / rhi, lhi / rlo
        print(f"  {name:<8} {bpe:>4} {lm:>8.1f} [{llo:.0f}-{lhi:.0f}]"
              f"  {refkey:>7} {rm:>7.1f} [{rlo:.0f}-{rhi:.0f}]"
              f"  {lm / rm:>6.1%} [{lo:.0%}-{hi:.0%}]")
        out.append({"kernel": name, "bytes_per_element": bpe, "reference": refkey,
                    "lyth_gbs": rows[name]["lyth"], "ref_gbs": rows[name]["ref"],
                    "lyth_median": lm, "ref_median": rm, "ratio": lm / rm,
                    "ratio_lo": lo, "ratio_hi": hi})
    print("\n  `split` writes two buffers and reads one; `triad*` is 2-read/1-write and does not")
    print("  describe it. The row is shown so the kernel is not missing, not because it matches.")

    if args.out:
        args.out.write_text(json.dumps(
            {"n": n, "rounds": args.rounds, "reps": args.reps,
             "device": torch.cuda.get_device_name(0), "rows": out}, indent=2) + "\n",
            encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
