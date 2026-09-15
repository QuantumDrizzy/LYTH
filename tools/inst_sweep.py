#!/usr/bin/env python3
"""Separate what the grid-stride loop costs from what the body costs, in instructions.

ADR-0012 measured the loop in **time** and found nothing: at the default grid, grid-stride
cannot be told apart from one element per thread. That is a true answer to "is it faster" and
no answer at all to "what does it cost", because a memory-bound kernel hides arithmetic.

This measures the cost directly. Per iteration the loop handles exactly one element, so its
per-element cost cannot depend on the grid; what the grid changes is how many elements each
thread's one-time setup is spread over. That gives an affine model:

    thread_instructions = threads * S  +  n * L

with ``S`` the per-thread setup and ``L`` the per-element cost of body plus loop. Both are
solved from measurements at two grids and then **checked against a third**, which is the point:
an affine model that predicts an unseen point to the instruction is a model, and one that does
not is a story.

``L`` is shared work plus body work, so a single kernel cannot say how much of it is the loop.
Running several kernels does: the loop is identical in all of them, so the smallest ``L`` bounds
it from above, and the differences between kernels are the bodies.

**[KNOWN LIMIT] These are SASS instructions and the compiler derives PTX.** ptxas reorders,
folds and selects its own instructions, so a PTX-level count is not expected to match and is not
compared here. What is measured is what the machine actually executed.
"""

import argparse
import csv
import io
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
METRIC = "smsp__thread_inst_executed.sum"


def find_ncu():
    """Locate ncu, which is not on PATH in this environment."""
    for base in sorted(pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"),
                       reverse=True):
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found under 'C:/Program Files/NVIDIA Corporation/Nsight Compute*'")


def measure(ncu, binary, source, machine, n, grid, block):
    """Thread-instructions executed by one launch at this grid."""
    cmd = [
        str(ncu), "--metrics", METRIC, "--csv",
        str(binary), "run", str(source),
        "--machine", str(machine), "-n", str(n), "--grid", str(grid),
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    rows = [r for r in csv.DictReader(io.StringIO(
        done.stdout[done.stdout.index('"ID","Process ID"'):]))
        if r.get("Metric Name") == METRIC]
    if len(rows) != 1:
        sys.exit(f"grid {grid}: expected one profiled launch, got {len(rows)}\n{done.stdout}")
    # ncu writes the value in the host locale, so 14.184.448 is one number and not three.
    return int(re.sub(r"[^0-9]", "", rows[0]["Metric Value"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", type=pathlib.Path,
                    default=[REPO / "examples" / f for f in
                             ("sum.lyth", "max.lyth", "saxpy.lyth", "horner.lyth")])
    ap.add_argument("--machine", type=pathlib.Path,
                    default=REPO / "fixtures/machine/sm_120.json")
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("-n", type=int, default=1048576)
    args = ap.parse_args()

    ncu = find_ncu()
    block = 256
    one_per_thread = -(-args.n // block)
    # Two to solve, the rest to falsify. The check grids are deliberately far from the solve
    # grids: a model fitted at the ends that also predicts the middle is not fitting noise.
    solve = [36, one_per_thread]
    check = [144, 576]

    print(f"n = {args.n}, block = {block}, one element per thread at grid {one_per_thread}")
    print(f"solving S and L at grids {solve}, checking at {check}\n")
    print(f"  {'kernel':<16} {'S/thread':>9} {'L/element':>10} {'check':>22} "
          f"{'1/thread':>9} {'default':>9}")
    print(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*22} {'-'*9} {'-'*9}")

    for src in args.sources:
        pts = {g: measure(ncu, args.binary, src, args.machine, args.n, g, block)
               for g in solve + check}
        (g0, g1) = solve
        t0, t1 = g0 * block, g1 * block
        s = (pts[g1] - pts[g0]) / (t1 - t0)
        l = (pts[g0] - t0 * s) / args.n

        worst = 0
        for g in check:
            want = g * block * s + args.n * l
            worst = max(worst, abs(want - pts[g]))
        verdict = "exact" if worst < 0.5 else f"off by {worst:.0f}"

        # What the two shapes cost per element. One element per thread pays the setup once
        # for every element; grid-stride spreads it over however many that thread handles.
        per_elem_1 = l + s
        default_grid = min(36 * 4, one_per_thread)
        per_elem_d = l + s * default_grid * block / args.n

        print(f"  {src.stem:<16} {s:9.4f} {l:10.4f} {verdict:>22} "
              f"{per_elem_1:9.2f} {per_elem_d:9.2f}")

    print("\n  S and L are thread-instructions. `1/thread` and `default` are instructions per")
    print("  element under the two launch shapes -- the quantity ADR-0012 could not see in time.")


if __name__ == "__main__":
    main()
