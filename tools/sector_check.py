#!/usr/bin/env python3
"""The falsification ADR-0015 pre-registered: does a strided access cost a whole sector?

The cost model charges a coalesced access 4 bytes per element and a strided one 32, and
`examples/transpose.lyth` is the first kernel this language can write that tells them apart.
This runs it under `ncu` and compares.

**It measures a control as well, and that is not optional.** `examples/copy2d.lyth` is the same
kernel -- rank 2, declared space, explicit indices, same extents, same payload -- with both
buffers walked the same way, which the model says is coalesced on both sides. Without it a
number from the transpose cannot be told apart from a number about rank-2 kernels at that size.

Three counters, because they answer different questions:

* ``lts__t_bytes.sum`` is what the L1s asked the L2 for. This is what a sector model is about.
* ``dram__bytes_op_read.sum`` and ``..._write.sum`` are what crossed the memory controller,
  which is what the roofline is about, and which the L2 is free to change.

The size sweep is the point rather than a single number: whether the two agree depends on
whether the working set fits in L2, and L2 capacity is read from the driver rather than assumed.
"""

import argparse
import csv
import io
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
METRICS = "dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum"


def find_ncu():
    for base in sorted(
        pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"), reverse=True
    ):
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found under 'C:/Program Files/NVIDIA Corporation/Nsight Compute*'")


def measure(ncu, binary, machine, source, size):
    cmd = [
        str(ncu), "--metrics", METRICS, "--csv",
        str(binary), "run", str(source), "--machine", str(machine),
        "--set", f"rows={size}", "--set", f"cols={size}",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if '"ID","Process ID"' not in done.stdout:
        sys.exit(f"{source.name} at {size}: no profile\n{done.stdout}\n{done.stderr}")
    rows = csv.DictReader(io.StringIO(done.stdout[done.stdout.index('"ID","Process ID"'):]))
    out = {}
    for r in rows:
        name = r.get("Metric Name")
        if name in METRICS:
            # ncu writes the value in the host locale, so 654.398.784 is one number.
            out[name] = int(re.sub(r"[^0-9]", "", r["Metric Value"]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--machine", type=pathlib.Path, default=REPO / "fixtures/machine/sm_120.json")
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    args = ap.parse_args()

    ncu = find_ncu()
    # name, derived bytes per element at the bus, as the cost model reports them
    kernels = [("copy2d.lyth", 8.0), ("transpose.lyth", 36.0)]

    for name, model in kernels:
        src = REPO / "examples" / name
        print(f"\n{name}: the model says {model:.0f} bytes per element at the bus")
        print(f"  {'size':>11} {'L2/elem':>9} {'vs model':>9} "
              f"{'DRAM rd':>8} {'DRAM wr':>8} {'DRAM tot':>9} {'vs payload':>10} {'ws':>8}")
        print(f"  {'-'*11} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*8}")
        for size in args.sizes:
            m = measure(ncu, args.binary, args.machine, src, size)
            n = size * size
            l2 = m["lts__t_bytes.sum"] / n
            rd = m["dram__bytes_op_read.sum"] / n
            wr = m["dram__bytes_op_write.sum"] / n
            ws = n * 8 / 1e6
            print(
                f"  {size:>5}x{size:<5} {l2:9.2f} {l2 / model:8.2f}x "
                f"{rd:8.2f} {wr:8.2f} {rd + wr:9.2f} {(rd + wr) / 8.0:9.2f}x {ws:7.0f}MB"
            )

    print("\n  `vs payload` is against 8 bytes per element, which both kernels ask for.")
    print("  A ratio below 1 means the L2 kept the writes: they never reached DRAM at all.")


if __name__ == "__main__":
    main()
