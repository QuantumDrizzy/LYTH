#!/usr/bin/env python3
"""Sweep the launch grid and ask whether grid-stride bought anything.

ADR-0012 names a falsification and then failed to test it:

    If grid-stride does not change achieved bandwidth against one-element-per-thread, the loop
    bought nothing for performance and its justification is the measurement it enables.

One-element-per-thread is recoverable: at ``--grid ceil(n / block)`` every thread runs the loop
exactly once, which is the old shape plus a loop head it executes twice. So the comparison is a
sweep with that point in it.

METHOD, and the part that is not obvious:

**Passes are interleaved.** Running every repetition of one grid, then every repetition of the
next, attributes a thermal drift or a clock-boost transition to whichever grid happened to be
measured late. Each pass here visits every grid once, and the per-grid figure is the median
across passes. That turns a drift into noise shared by all points instead of a fake result at
one of them.

Both spreads are reported: ``within`` is the run-to-run spread inside a single timing call,
``across`` is the spread of pass medians. **If ``across`` is larger than the differences between
grids, the experiment cannot distinguish them, and that is the answer** -- not a reason to pick
the smallest number and call it a win.
"""

import argparse
import json
import pathlib
import subprocess
import statistics
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent


def spread(xs):
    """(max - min) / median, as a fraction."""
    m = statistics.median(xs)
    return (max(xs) - min(xs)) / m if m else 0.0


def one(binary, source, machine, n, grid, reps, sets):
    """Run one timing and return its JSON record."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "t.json"
        cmd = [
            str(binary), "run", str(source),
            "--machine", str(machine),
            "-n", str(n),
            "--grid", str(grid),
            "--time", str(reps),
            "--json", str(out),
        ]
        for kv in sets:
            cmd += ["--set", kv]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            sys.exit(f"grid {grid} failed:\n{done.stdout}\n{done.stderr}")
        return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("--machine", type=pathlib.Path,
                    default=REPO / "fixtures/machine/sm_120.json")
    ap.add_argument("--binary", type=pathlib.Path,
                    default=REPO / "target/release/lyth.exe")
    ap.add_argument("-n", type=int, default=67108864,
                    help="elements; the default is 256 MB per buffer, far past L2")
    ap.add_argument("--reps", type=int, default=7, help="timed runs inside one pass")
    ap.add_argument("--passes", type=int, default=5,
                    help="interleaved passes over every grid")
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--out", type=pathlib.Path, help="write the raw records as JSON")
    args = ap.parse_args()

    if not args.binary.exists():
        sys.exit(f"{args.binary} not found -- build it with `cargo build --release -p lyth`")

    block = 256
    one_per_thread = -(-args.n // block)  # ceil: every thread runs the loop exactly once
    grids = sorted({1, 9, 18, 36, 72, 144, 288, 576, 1152, 4608, one_per_thread})

    print(f"sweep {args.source.name}  n={args.n}  block={block}")
    print(f"  {args.passes} interleaved passes x {args.reps} timed runs per grid")
    print(f"  grid {one_per_thread} is one element per thread: the shape grid-stride replaced")
    print()

    # The device's SM count, so the default grid can be named. Taken from the tool rather than
    # assumed, for the same reason the tool takes it from the driver.
    probe = subprocess.run(
        [str(args.binary), "run", str(args.source), "--machine", str(args.machine),
         "-n", "1024", "--grid", "1"] + [x for kv in args.sets for x in ("--set", kv)],
        capture_output=True, text=True)
    sm_count = 0
    for line in probe.stdout.splitlines():
        if "SMs x" in line:
            sm_count = int(line.split("(")[1].split(" SMs")[0])
    waves = 4

    records = {g: [] for g in grids}
    for p in range(args.passes):
        print(f"  pass {p + 1}/{args.passes}", end="", flush=True)
        for g in grids:
            records[g].append(one(args.binary, args.source, args.machine,
                                  args.n, g, args.reps, args.sets))
            print(".", end="", flush=True)
        print()
    print()

    baseline = records[grids[0]][0].get("baseline_gbs")
    print(f"  {'grid':>8}  {'GB/s':>8}  {'ms':>9}  {'within':>7}  {'across':>7}  {'vs base':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*8}")

    rows = []
    for g in grids:
        mss = [r["ms_median"] for r in records[g]]
        ms = statistics.median(mss)
        gbs = statistics.median([r["achieved_gbs"] for r in records[g]])
        within = statistics.median([r["ms_spread"] for r in records[g]])
        across = spread(mss)
        rows.append((g, gbs, ms, within, across))
        tag = "  <- one per thread" if g == one_per_thread else ""
        base = f"{gbs / baseline * 100:7.1f}%" if baseline else "      --"
        print(f"  {g:>8}  {gbs:8.2f}  {ms:9.4f}  {within:6.1%}  {across:6.1%}  {base}{tag}")

    # The question is not "which grid is fastest" -- if the fastest happens to be the
    # one-per-thread point, comparing the best against it is comparing it with itself and
    # always yields zero. The question is whether the GRID-STRIDE regime, at the grid the tool
    # actually picks by default, matches the shape it replaced.
    sm_default = min(sm_count * waves, one_per_thread) if sm_count else None
    ref = next(r for r in rows if r[0] == one_per_thread)
    striding = [r for r in rows if r[0] != one_per_thread]
    best_stride = max(striding, key=lambda r: r[1])
    default = next((r for r in rows if r[0] == sm_default), best_stride)

    def compare(label, row):
        gain = row[1] / ref[1] - 1.0
        # Noise for THIS comparison: the two points involved, not the worst point in the sweep
        # (which is grid 1, where a low-occupancy launch is naturally jittery).
        noise = max(row[4], ref[4])
        verdict = "inside the noise" if abs(gain) <= noise else "outside the noise"
        print(f"  {label:<28} {row[1]:7.2f} GB/s   {gain:+6.1%} vs one-per-thread"
              f"   ({verdict}, +/-{noise:.1%})")
        return gain, noise

    print(f"  one element per thread (grid {ref[0]}) at {ref[1]:.2f} GB/s is the reference")
    print()
    d_gain, d_noise = compare(f"default grid {default[0]}", default)
    if best_stride[0] != default[0]:
        compare(f"best grid-stride {best_stride[0]}", best_stride)

    # Where the curve stops paying: the first grid within 5% of the plateau.
    plateau = max(r[1] for r in rows)
    knee = next(r for r in rows if r[1] >= 0.95 * plateau)
    print()
    print(f"  saturates at grid {knee[0]} ({knee[0] / sm_count:.0f} blocks/SM) "
          f"-- {knee[1] / plateau:.0%} of the plateau")
    print()
    if abs(d_gain) <= d_noise:
        print("  VERDICT: at the default grid the two shapes cannot be told apart. Grid-stride")
        print("  is not shown to be faster, and its justification is the measurement it makes")
        print("  possible, as ADR-0012 allowed.")
    elif d_gain < 0:
        print(f"  VERDICT: grid-stride at the default grid is {-d_gain:.1%} SLOWER than the")
        print("  shape it replaced, and the gap is outside the spread of both points. The loop")
        print("  did not buy speed; it cost some. Fewer threads in flight is less memory-level")
        print("  parallelism, which is the expected direction for a memory-bound kernel.")
    else:
        print(f"  VERDICT: grid-stride at the default grid is {d_gain:.1%} faster, outside the")
        print("  spread of both points. Not a like-for-like kernel comparison: both shapes are")
        print("  this compiler's output, not hand-written CUDA.")

    if args.out:
        args.out.write_text(json.dumps(
            {"n": args.n, "block": block, "passes": args.passes, "reps": args.reps,
             "one_per_thread_grid": one_per_thread,
             "records": {str(g): records[g] for g in grids}},
            indent=2) + "\n")
        print(f"\n  raw records -> {args.out}")


if __name__ == "__main__":
    main()
