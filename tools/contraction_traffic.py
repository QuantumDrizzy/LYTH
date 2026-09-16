#!/usr/bin/env python3
"""The falsification ADR-0018 pre-registered: does a tiled contraction move `4 * (2k/T + 1)`?

The compiler derives a contraction's traffic as an **expression** in the contracted extent --
`0.25 * k + 4` bytes per output at `tile 32, 32` -- and this asks the silicon whether that is
what moved.

**Nothing here writes the model.** The coefficients are read out of the manifest the compiler
emitted, so the number being checked comes from the compiler that generated the kernel and no
human can quietly adjust it. Same discipline as ADR-0009 and `sector_check.py`.

**The control is the tile, and it is not optional.** The same source at `tile 16, 16` derives
`0.5 * k + 4`, twice the traffic, from the same body, the same buffers and the same extents --
one number changed in a declaration. Without it, a figure that matches at `T = 32` cannot be
told apart from a figure that is about matmuls at that size. With it, the measurement is about
the *schedule*, which is the whole claim of ADR-0018.

Three counters, because they answer different questions:

* ``lts__t_bytes.sum`` is what the L1s asked the L2 for. **This is what the model claims**, and
  ADR-0015 measured the sector model exact at this interface and wrong at DRAM in both
  directions.
* ``dram__bytes_op_read.sum`` and ``..._write.sum`` are what crossed the memory controller.
  Pre-registered to come out **lower**, not equal: a block's `A` panel is shared by every block
  in its row of `C` and the L2 serves it. Writing "17 GB of DRAM" would have been the tidy
  sentence and false in the direction ADR-0015 already measured.

The size sweep is the point rather than a single number. Whether DRAM can serve the reuse
depends on whether the working set fits in L2, so the two figures are expected to diverge and
then converge as the problem outgrows the cache.
"""

import argparse
import csv
import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
L1_HIT = "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"
METRICS = ",".join(
    ["dram__bytes_op_read.sum", "dram__bytes_op_write.sum", "lts__t_bytes.sum", L1_HIT]
)

# The tolerance ADR-0017 earned rather than a round number. An untiled transpose overshot its
# sector model by read-for-ownership; a matmul writes `C` once and coalesced, so the same
# mechanism should not appear. If measured differs from derived by more than this, **the
# difference is the result** and gets its own investigation -- the tolerance does not widen.
#
# The tolerance is against the **L1-adjusted** figure, for the reason in the loop below: a gap
# that the L1 hit rate accounts for to within 0.15 points is a gap that is explained, and a
# tolerance that ignored the explanation would either fail a correct model or hide a real one.
#
# **Both directions.** The first version of this script compared `max(excess)` and so waved
# through a 5.4% *undershoot* at `tile 16, 16` without a word. A model that over-states traffic
# is safer than one that under-states it and is exactly as wrong, and ADR-0015's falsification
# table already recorded the DRAM figure as wrong in both directions. Checking one direction is
# how a cost model acquires a silent bias.
TOLERANCE = 0.02


def find_ncu():
    for base in sorted(
        pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"), reverse=True
    ):
        for cand in (base / "target").glob("*/ncu.exe"):
            return cand
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found under 'C:/Program Files/NVIDIA Corporation/Nsight Compute*'")


def coarsening_for(tile: int, cap: int = 1024) -> int:
    """The per-axis factor that brings `tile x tile` threads under the block cap.

    Not a schedule search. A tile puts one thread on each of its elements, the device caps a
    block at `cap`, and `coarsen` says how many of those elements one thread owns -- so the
    smallest legal factor is fully determined by the tile and the cap, and every other choice
    is a larger one. Powers of two only, which is the parser's rule (ADR-0021): a thread's
    position in the tile stays a shift and a mask.

    Returns 1 when the tile already fits, which is the absent declaration.
    """
    c = 1
    while (tile // c) ** 2 > cap:
        c *= 2
        if tile % c:
            sys.exit(f"tile {tile} needs a coarsening its edge does not divide")
    return c


def variant(tile: int, out: pathlib.Path) -> pathlib.Path:
    """The matmul example with its tile, its coarsening and its declared limit rewritten.

    The declaration has to move with the tile: the asymptote is `T/4`, so a source that kept
    `intensity asymptotic 8.0` at `tile 16, 16` would be refused -- which is the compiler doing
    its job and not something to work around.

    Past `tile 32` the source also needs `coarsen`, because 64 x 64 is 4096 threads and no block
    is that wide (ADR-0021). **That line changes nothing the cost model derives**, which is why
    it belongs in a measurement rather than in an argument: if the derivation is right, the
    traffic halves from `tile 32` to `tile 64` and the coarsening is invisible in it.
    """
    src = (REPO / "examples/matmul.lyth").read_text(encoding="utf-8")
    c = coarsening_for(tile)
    tile_line = f"    tile {tile}, {tile}\n"
    if c > 1:
        tile_line += f"    coarsen {c}, {c}\n"
    src = src.replace("    tile 32, 32\n", tile_line)
    src = src.replace(
        "    intensity asymptotic 8.0\n", f"    intensity asymptotic {tile / 4}\n"
    )
    path = out / f"matmul_t{tile}.lyth"
    path.write_text(src, encoding="utf-8")
    return path


def derived(binary, machine, source, out: pathlib.Path):
    """The traffic expression, read from the manifest the compiler wrote."""
    man = out / (source.stem + ".json")
    done = subprocess.run(
        [str(binary), "build", str(source), "--machine", str(machine),
         "-o", "nul" if sys.platform == "win32" else "/dev/null", "--manifest", str(man)],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        sys.exit(f"{source.name} did not compile:\n{done.stdout}\n{done.stderr}")
    sym = json.loads(man.read_text(encoding="utf-8"))["contract"]["symbolic"]
    return sym


def measure(ncu, binary, machine, source, m, n, k):
    cmd = [
        str(ncu), "--metrics", METRICS, "--csv",
        str(binary), "run", str(source), "--machine", str(machine),
        "--set", f"m={m}", "--set", f"n={n}", "--set", f"k={k}",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if '"ID","Process ID"' not in done.stdout:
        sys.exit(f"{source.name} at {m}x{n}x{k}: no profile\n{done.stdout}\n{done.stderr}")
    if "BIT-EXACT" not in done.stdout:
        # A traffic number from a kernel that computes the wrong thing is a number about
        # nothing. `sector_check.py` learned this the same way.
        sys.exit(f"{source.name} at {m}x{n}x{k} is not bit-exact; traffic would mean nothing")
    body = done.stdout[done.stdout.index('"ID","Process ID"'):]
    out = {}
    for r in csv.DictReader(io.StringIO(body)):
        name = r.get("Metric Name")
        if name in METRICS:
            # ncu writes the value in the host locale: `273.274.400` is one integer and
            # `5,52` is one ratio. Two different separators in one output.
            raw = r["Metric Value"]
            if name.endswith(".pct"):
                out[name] = float(raw.replace(".", "").replace(",", "."))
            else:
                out[name] = int(re.sub(r"[^0-9]", "", raw))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--machine", type=pathlib.Path, default=REPO / "fixtures/machine/sm_120.json")
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--tiles", type=int, nargs="+", default=[64, 32, 16])
    args = ap.parse_args()

    ncu = find_ncu()
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for tile in args.tiles:
            src = variant(tile, tmp)
            c = coarsening_for(tile)
            sym = derived(args.binary, args.machine, src, tmp)
            label = f"tile {tile} x {tile}"
            if c > 1:
                label += f" + coarsen {c}, {c}"
            label += f"  [{(tile // c) ** 2} threads/block]"
            print(f"\n{label}: the compiler derives {sym['bytes']} bytes per output")
            print(f"  asymptotic intensity {sym['asymptotic_intensity']:.4f} = T/4 = {tile / 4}")
            print(
                f"  {'m=n=k':>7} {'derived':>9} {'L2/out':>9} {'vs model':>9} "
                f"{'L1 hit':>8} {'DRAM/out':>9} {'DRAM vs A+B':>12}"
            )
            for size in args.sizes:
                got = measure(ncu, args.binary, args.machine, src, size, size, size)
                outputs = size * size
                want = sym["bytes_per_extent"] * size + sym["bytes_fixed"]
                l2 = got["lts__t_bytes.sum"] / outputs
                dram = (
                    got["dram__bytes_op_read.sum"] + got["dram__bytes_op_write.sum"]
                ) / outputs
                excess = l2 / want - 1.0
                # What the model claims, once the L1 has been given credit for what it served.
                adjusted = excess + got[L1_HIT] / 100.0
                if abs(adjusted) > abs(worst):
                    worst = adjusted
                # What DRAM would move if it read each input matrix exactly once and the L2
                # served every re-read. The lower bound this schedule can reach at all.
                floor = 2 * size * size * 4 / outputs
                # The L1 hit rate is not decoration: the derived figure is what the
                # **kernel** asks for and `lts__t_bytes` is what the **L1** asks the L2 for,
                # so they differ by exactly what the L1 served. Every kernel in this language
                # before a contraction had a hit rate of 0 -- streaming, no block re-reads
                # another block's data -- so the two were the same number and the distinction
                # never had to be made. Measured on the matmul: 0.00% / +0.01%,
                # 2.98% / -2.89%, 5.52% / -5.37%.
                print(
                    f"  {size:>7} {want:>9.1f} {l2:>9.2f} {excess:>+8.2%} "
                    f"{got[L1_HIT]:>7.2f}% {dram:>9.2f} {dram / floor:>11.2f}x"
                )

    print()
    if abs(worst) <= TOLERANCE:
        print(
            f"HOLDS: the largest disagreement is {worst:+.2%} once the L1 hit rate is "
            f"credited, inside +/-{TOLERANCE:.0%}"
        )
    else:
        print(
            f"EXCEEDED: {worst:+.2%} against a +/-{TOLERANCE:.0%} tolerance.\n"
            "The difference is the result. Do not widen this number -- find the mechanism."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
