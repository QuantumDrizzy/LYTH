#!/usr/bin/env python3
"""The same kernels at four bytes and at two. ADR-0024 step 3.

    python bench/narrow_vs_wide.py

**Do not pipe this into `tail`.** It exits 2 when a kernel misses its pre-registered band, and
a shell pipeline reports the last command's status -- so `| tail` turns every failure into a
zero. That is not hypothetical: it happened on the step-4 run and looked, for a minute, like a
check whose failure state was unreachable.

This is the measurement ADR-0024 exists for, and it is worth having because **the model
predicts two different answers to one change**:

| | f32 -> f16 |
|---|---|
| bytes moved at DRAM | halved |
| shared-memory **accesses** | unchanged -- one per operand per term either way |

ADR-0022 measured that a tiled contraction is paced by the shared pipe, which is priced in
accesses and not in bytes: 1530 G accesses/s, with a broadcast and a coalesced read agreeing to
0.2% while their payloads differ by 32x. So:

> `saxpy` should take about **half** the time. The matmul should take about **the same**.

Both come out of one model, from different levels of it. A result where both halve is a result
where ADR-0022's `smem binds` line is decoration. A result where neither moves is a result where
the traffic model was never about time.

Three precisions the ADR pre-registered, each of which is a way to read this wrong:

* **The claim is time, not GB/s.** The achievable bandwidth does not change; the bytes do. This
  prints milliseconds and the GB/s is deliberately absent.
* **"Unchanged" means inside +/-5%, not exactly zero.** That band is the measured run-to-run
  reproducibility of this project's timing harness -- 1.6% across two guarded nine-round runs --
  plus room for thermal drift. It is **not** ADR-0022's +/-0.80%, which is how closely the
  *traffic* model matched `ncu`: borrowing a cheap quantity's uncertainty into an expensive
  measurement manufactures falsifications out of ordinary noise (ADR-0000).
* **Sizes are chosen to exceed the L2.** At `n = 4M` a saxpy's two buffers are 33.6 MB against a
  34 MB L2, and it reports 1645 GB/s -- four times this device's DRAM bandwidth, because it
  never went to DRAM. A ratio measured out of cache would be a ratio about cache.

One refinement the emitter forced, recorded before the run rather than after: narrowing adds a
`cvt.f32.f16` per shared load. If the shared pipe really is the constraint the conversion is
free; if it is not, **the matmul gets slower**. So the matmul prediction is "not faster", with
"slower by the cvt" as the interesting way for it to be interesting.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from gpu_health import Watch  # noqa: E402

# (example, extra args, reps, what the model says about halving the element)
#
# `--tol 1.0` throughout: the declared `intensity` is the f32 one and halving the element
# doubles it, which the compiler correctly refuses. That refusal is tested in
# `lyth-lang/tests/element_width.rs`; conflating the contract with the measurement would test
# neither.
PLAN = [
    ("saxpy.lyth", ["-n", "67108864"], 30, "dram", 2.0),
    ("sum.lyth", ["-n", "67108864"], 30, "dram", 2.0),
    ("axpby.lyth", ["-n", "67108864"], 30, "dram", 2.0),
    ("transpose-tiled.lyth", ["--set", "rows=4096", "--set", "cols=4096"], 30, "dram", 2.0),
    ("matmul.lyth", ["--set", "m=1024", "--set", "n=1024", "--set", "k=1024"], 50, "smem", 1.0),
    ("matmul-coarse.lyth",
     ["--set", "m=1024", "--set", "n=1024", "--set", "k=1024"], 50, "smem", 1.0),
]

# ADR-0024 step 4. bf16 is here because the traffic model **cannot tell it from f16** -- two
# bytes is two bytes -- which is a prediction rather than an oversight: same time, different
# error. Measuring it in the same table as the time keeps the model's blind spot beside the
# thing it gets right.
WIDTHS = ("f32", "f16", "bf16")


def measure(binary: pathlib.Path, machine: pathlib.Path, example: str, width: str,
            extra: list[str], reps: int, tmp: pathlib.Path) -> dict | None:
    src = (REPO / "examples" / example).read_text(encoding="utf-8").replace(
        "[f32;", f"[{width};"
    )
    f = tmp / f"{width}_{example}"
    f.write_text(src, encoding="utf-8")
    out = tmp / f"{width}_{example}.json"

    done = subprocess.run(
        [str(binary), "run", str(f), "--machine", str(machine),
         "--time", str(reps), "--json", str(out), "--tol", "1.0", *extra],
        capture_output=True, text=True,
    )
    text = done.stdout + done.stderr
    # A timing from a kernel that computes the wrong thing is a number about nothing. Every
    # run here verifies at the size it is timed at, which is why the matmul is 1024 cubed and
    # not ADR-0022's 2048: the host oracle is O(n^3) in an interpreter and 2048 costs a quarter
    # of an hour per point. Timing a size the oracle never checked would be cheaper and would
    # be trusting the emitter at exactly the moment it changed.
    if "BIT-EXACT" not in text:
        print(f"    {example} {width}: NOT BIT-EXACT -- refusing to time it\n{text[-400:]}",
              flush=True)
        return None
    if not out.exists():
        print(f"    {example} {width}: no timing\n{text[-400:]}", flush=True)
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--machine", type=pathlib.Path,
                    default=REPO / "fixtures/machine/sm_120.json")
    args = ap.parse_args()

    watch = Watch().start()
    print(f"\n  {'kernel':<22} {'binds':<6} {'f32 ms':>9} {'f16 ms':>9} {'bf16 ms':>9} "
          f"{'f32/f16':>8} {'bf16/f16':>9} {'time':>7} {'twin':>7}", flush=True)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for example, extra, reps, binds, predicted in PLAN:
            got = {}
            for w in WIDTHS:
                r = measure(args.binary, args.machine, example, w, extra, reps, tmp)
                if r is None:
                    break
                got[w] = r
            if len(got) != len(WIDTHS):
                continue
            a = got["f32"]["ms_median"]
            b, c = got["f16"]["ms_median"], got["bf16"]["ms_median"]
            ratio = a / b
            # "About 2x" and "about 1x" are the same +/-5% band, applied to different centres.
            ok = abs(ratio / predicted - 1.0) <= 0.05
            # Step 4's claim: the traffic model derives the same bytes for both two-byte types,
            # so it predicts the same time. Same band, because it is the same instrument.
            twin = abs(c / b - 1.0) <= 0.05
            rows.append((example, binds, a, b, c, ratio, predicted, ok, twin))
            print(f"  {example:<22} {binds:<6} {a:>9.4f} {b:>9.4f} {c:>9.4f} {ratio:>8.3f} "
                  f"{c / b:>9.3f} {'ok' if ok else 'MISSED':>7} {'ok' if twin else 'MISSED':>7}",
                  flush=True)

    print(flush=True)
    for label, want in (("dram", 2.0), ("smem", 1.0)):
        sel = [r for r in rows if r[1] == label]
        if sel:
            v = [r[5] for r in sel]
            print(f"  {label}-bound: {len(sel)} kernels, f32/f16 {min(v):.2f}-{max(v):.2f}x "
                  f"(predicted {want})", flush=True)
    if rows:
        t = [r[4] / r[3] for r in rows]
        print(f"  bf16 against f16, all {len(rows)}: {min(t):.3f}-{max(t):.3f}x -- the traffic "
              f"model derives the same bytes for both, so it predicts 1.0", flush=True)

    print(flush=True)
    watch.report()
    if watch.clean is False:
        return 1
    return 0 if all(r[7] and r[8] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
