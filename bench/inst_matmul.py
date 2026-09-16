#!/usr/bin/env python3
"""What each compiler's matmul actually executes. ADR-0021 step 5.

    python bench/inst_matmul.py

Drives `ncu` over `vs_handwritten_matmul.py --once`, which launches each of the six kernels
exactly once and exits, and reads the counters back. Six launches in a fixed order, so the CSV
rows map to kernels by position -- the two nvrtc builds of a tile share an entry point name and
cannot be told apart any other way.

The counters, and what each one is here to settle (ADR-0021 pre-registered all three):

* ``smsp__thread_inst_executed.sum`` -- **the ADR-0020 quantity.** LYTH emitted 17% more of
  these than nvcc on a saxpy and lost nothing, because that kernel waited on memory. This is the
  first kernel dense enough for the difference to matter.
* ``dram__bytes.sum`` -- prediction 1: at 2048 the operands fit in a 34 MB L2, so DRAM should
  move each matrix once and be nowhere near the constraint the printed roofline assumes.
* ``lts__t_bytes.sum`` -- the figure the cost model derives, checked independently in step 4.
* ``smsp__inst_executed_op_shared_ld.sum`` -- shared-memory loads, the third quantity coarsening
  halves. Listed so the ADR cannot quietly credit the one it models.

Duration is deliberately **not** collected. Under ncu a kernel is serialised and replayed, so
its reported time is not this kernel's time; the throughput figures come from
`vs_handwritten_matmul.py` with CUDA events and no profiler attached. Every counter here is an
integer count, which also removes a parsing hazard: ncu writes values in the host locale, where
`2.176.532.096` is one integer and `1.234` is ambiguous between one and one thousand two hundred
and thirty-four. Stripping non-digits is only safe because nothing here has a fractional part.
"""

from __future__ import annotations

import csv
import io
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

METRICS = [
    "smsp__thread_inst_executed.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
]

# The launch order `vs_handwritten_matmul.py` produces: per tile, LYTH then nvrtc --fmad=false
# then nvrtc default.
ORDER = [
    "LYTH tile 16",
    "nvrtc tile 16",
    "nvrtc tile 16 [fmad]",
    "LYTH tile 32",
    "nvrtc tile 32",
    "nvrtc tile 32 [fmad]",
    "LYTH tile 64 c2",
    "nvrtc tile 64 c2",
    "nvrtc tile 64 c2 [fmad]",
]


def find_ncu() -> pathlib.Path:
    for base in sorted(
        pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"), reverse=True
    ):
        for cand in (base / "target").glob("*/ncu.exe"):
            return cand
    sys.exit("ncu not found")


def main() -> int:
    ncu = find_ncu()
    cmd = [
        str(ncu), "--metrics", ",".join(METRICS), "-k", "regex:matmul", "--csv",
        sys.executable, str(HERE / "vs_handwritten_matmul.py"), "--once",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if '"ID","Process ID"' not in done.stdout:
        sys.exit(f"no profile:\n{done.stdout}\n{done.stderr}")
    body = done.stdout[done.stdout.index('"ID","Process ID"'):]

    got: dict[int, dict[str, float]] = {}
    names: dict[int, str] = {}
    for r in csv.DictReader(io.StringIO(body)):
        i = int(r["ID"])
        names[i] = r["Kernel Name"]
        metric = r.get("Metric Name")
        if metric in METRICS:
            # Every metric here is an integer count; see the note in the docstring.
            got.setdefault(i, {})[metric] = float(re.sub(r"[^0-9]", "", r["Metric Value"]))

    ids = sorted(got)
    if len(ids) != len(ORDER):
        sys.exit(f"expected {len(ORDER)} launches, profiled {len(ids)}: "
                 f"{[names[i] for i in ids]}")

    n = 2048
    outputs = n * n
    flops = 2.0 * n ** 3
    print(f"\n  m = n = k = {n}: {outputs:,} outputs, {flops / 1e9:.1f} GFLOP\n")
    print(f"  {'':<24} {'thread-inst':>14} {'per flop':>9} {'shared ld':>12} "
          f"{'DRAM MB':>9} {'L2 B/out':>9}")
    inst = {}
    for i, label in zip(ids, ORDER):
        g = got[i]
        ti = g["smsp__thread_inst_executed.sum"]
        inst[label] = ti
        print(f"  {label:<24} {ti:>14,.0f} {ti / flops:>9.2f} "
              f"{g['smsp__inst_executed_op_shared_ld.sum']:>12,.0f} "
              f"{g['dram__bytes.sum'] / 1e6:>9.1f} "
              f"{g['lts__t_bytes.sum'] / outputs:>9.2f}")

    print()
    for tile in ("tile 16", "tile 32", "tile 64 c2"):
        l, h = f"LYTH {tile}", f"nvrtc {tile}"
        print(f"  {tile:<12} LYTH / nvrtc instructions = {inst[l] / inst[h]:.1%}")
    print(f"  {'across tiles':<12} coarsening changes LYTH's instructions by "
          f"{inst['LYTH tile 64 c2'] / inst['LYTH tile 32']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
