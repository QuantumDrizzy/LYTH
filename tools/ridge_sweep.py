#!/usr/bin/env python3
"""Does the machine file's ridge predict where the kernel stops being bandwidth-bound?

`fixtures/machine/sm_120.json` states a peak of 15.37 TFLOP/s and 358.43 GB/s, so
`lyth-probe` derives a ridge of 42.88 flop/byte and every `lyth run` prints a verdict against
it. **Nothing has ever tested that verdict.** Every kernel this compiler had compiled sat below
0.75 flop/byte, so `Regime::of` had only ever returned `memory-bound`: a classifier with one
observed output is an opinion with a type signature.

A Horner polynomial is the instrument. Each extra degree is one `fma` -- two flops -- and moves
no bytes, so degree sweeps intensity from 0.25 upward with the traffic held at 8 bytes per
element. Intensity is exactly `degree / 4`.

The prediction, made by the machine file before any of this is run:

* below the ridge the kernel is bandwidth-bound, so **time per element is flat** and achieved
  TFLOP/s climbs linearly with degree;
* above it the kernel is compute-bound, so **TFLOP/s flattens at peak** and achieved bandwidth
  falls away.

The knee between those two lines is the ridge the machine actually has. It is compared with the
ridge the file claims, and the gap is the result -- in either direction.

Both ceilings are measured figures from the same file, so this is measured against measured.
Comparing an achieved rate with a datasheet peak is the mistake that makes every roofline look
flattering, and ADR-0004 already refuses the field that would let it happen here.

**Passes are interleaved**, for the reason `grid_sweep.py` gives: visiting every degree once per
pass turns a thermal drift into noise shared by all points rather than a fake knee at whichever
degree was measured last.
"""

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent


def source(degree):
    """A chain of `degree` fused multiply-adds that stays finite.

    The obvious form is a Horner polynomial in `x`, `y = (y * x + c)` repeated. It is the wrong
    instrument here: the input generator produces `x` in [-4, 4), so for three quarters of the
    elements the recurrence diverges, and by degree 400 **68.8% of them are +-inf**. The launch
    still retires every fma and the timing is still a timing, but the bit-exact check
    degenerates into comparing infinity with infinity, and a verification that cannot fail is
    not a verification.

    `y = (y * c + x)` with `c = 0.5` converges to `2x` instead. Same instruction, same count,
    same byte traffic, same derived intensity -- and every value finite and below 8, so the
    check still has something to check. The multiplier is the scalar and the addend is the
    streamed value, which also keeps `x` live in every step so nothing can be folded away.
    """
    e = "x"
    for _ in range(degree):
        e = f"({e} * c + x)"
    return (
        "machine sm_120\n\n"
        f"kernel poly{degree}(n: u32, c: f32, x: [f32], y: [f32])\n"
        "    stream x : dram -> reg\n"
        "    stream y : dram -> reg, drain\n\n"
        "    at reg:\n"
        f"        y = {e}\n"
    )


def run(binary, machine, degree, n, reps, tmp):
    src = tmp / f"poly{degree}.lyth"
    src.write_text(source(degree), encoding="utf-8", newline="\n")
    out = tmp / f"poly{degree}.json"
    done = subprocess.run(
        [str(binary), "run", str(src), "--machine", str(machine),
         "-n", str(n), "--time", str(reps), "--json", str(out), "--set", "c=0.5"],
        capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"degree {degree} failed:\n{done.stdout}\n{done.stderr}")
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", type=pathlib.Path,
                    default=REPO / "fixtures/machine/sm_120.json")
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("-n", type=int, default=16777216, help="elements; 64 MB per buffer")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    machine = json.loads(args.machine.read_text())
    peak_tflops = machine["peak_tflops"]
    bw = next(l["bandwidth_gbs"] for l in machine["levels"] if l["name"] == "dram")
    ridge = peak_tflops * 1e12 / (bw * 1e9)

    degrees = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 172, 192, 224, 256, 320, 384, 448, 512]

    print(f"{args.machine.name}: {peak_tflops} TFLOP/s, {bw} GB/s")
    print(f"  so the file's ridge is {ridge:.2f} flop/byte, which is degree {ridge * 4:.0f}\n")
    print(f"  n = {args.n}, {args.passes} interleaved passes x {args.reps} timed runs\n")

    records = {d: [] for d in degrees}
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        for p in range(args.passes):
            print(f"  pass {p + 1}/{args.passes}", end="", flush=True)
            for d in degrees:
                records[d].append(run(args.binary, args.machine, d, args.n, args.reps, tmp))
                print(".", end="", flush=True)
            print()
    print()

    print(f"  {'degree':>6} {'flop/byte':>9} {'ms':>9} {'GB/s':>8} {'%bw':>6} "
          f"{'TFLOP/s':>8} {'%peak':>6}  regime")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*8} {'-'*6} {'-'*8} {'-'*6}  {'-'*12}")

    rows = []
    for d in degrees:
        ms = statistics.median(r["ms_median"] for r in records[d])
        gbs = statistics.median(r["achieved_gbs"] for r in records[d])
        intensity = d / 4.0
        # flops retired per second, from the derived flop count and the measured time.
        tflops = (2.0 * d * args.n) / (ms * 1e-3) / 1e12
        r = intensity / ridge
        regime = "memory-bound" if r < 0.5 else ("near ridge" if r <= 2.0 else "compute-bound")
        rows.append((d, intensity, ms, gbs, tflops))
        print(f"  {d:>6} {intensity:9.2f} {ms:9.3f} {gbs:8.2f} {gbs / bw:5.0%} "
              f"{tflops:8.2f} {tflops / peak_tflops:5.0%}  {regime}")

    # The empirical ridge: the intensity at which the two ceilings cross. Below it the kernel
    # is at bandwidth peak; above it at flop peak. Taking the best observed value of each
    # rather than the stated one answers "where does THIS kernel cross", which is the question
    # the verdict is really making a claim about.
    best_gbs = max(r[3] for r in rows)
    best_tflops = max(r[4] for r in rows)
    empirical = best_tflops * 1e12 / (best_gbs * 1e9)
    print()
    print(f"  best achieved bandwidth   {best_gbs:7.2f} GB/s   ({best_gbs / bw:.0%} of the file)")
    print(f"  best achieved throughput  {best_tflops:7.2f} TFLOP/s ({best_tflops / peak_tflops:.0%} of the file)")
    print(f"  empirical ridge           {empirical:7.2f} flop/byte")
    print(f"  file's ridge              {ridge:7.2f} flop/byte   ({empirical / ridge - 1:+.1%})")

    if args.out:
        args.out.write_text(json.dumps(
            {"n": args.n, "passes": args.passes, "reps": args.reps,
             "ridge_declared": ridge, "ridge_empirical": empirical,
             "records": {str(d): records[d] for d in degrees}}, indent=2) + "\n")
        print(f"\n  raw records -> {args.out}")


if __name__ == "__main__":
    main()
