#!/usr/bin/env python3
"""ADR-0030 Q4: with the emitter owning the store order, does `cu_q` as declared reach the L2 like
`cu_q_grouped` did?

    q4_store_order.py run      measure cu_q at the twelve P2 follow-up pairs
    q4_store_order.py analyze  compare with cu_q_grouped in docs/prereg/ADR-0029.p2-followup-results.json

Instruments are the P2 follow-up's (tools/p2_followup.py): ncu, default cache control, 3 launches in
shuffled rounds, median of lts__t_bytes.sum per quad. The pass is ADR-0030's, frozen: within 5% of
cu_q_grouped at every pair. The comparison is against the recorded values, measured under the
declared order, not against a re-run: after step 2 the two sources compile to the same stores.
"""

import argparse
import json
import pathlib
import random
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import p2_followup as F  # noqa: E402  the frozen instruments, reused unchanged

RESULTS = F.REPO / "docs/prereg/ADR-0030.q4-results.json"
BASELINE = F.REPO / "docs/prereg/ADR-0029.p2-followup-results.json"
SEED, TOL = 300, 0.05


def measure():
    if F.git("status", "--porcelain").strip():
        sys.exit("the git tree is dirty")
    ncu = F.find_ncu()
    rng = random.Random(SEED)
    runs = {F.key_of(("cu_q", p)): [] for p in F.PAIRS}
    for rnd in range(F.ROUNDS):
        order = F.PAIRS[:]
        rng.shuffle(order)
        for p in order:
            runs[F.key_of(("cu_q", p))].append(F.ncu_once(ncu, ("cu_q", p)))
        print(f"round {rnd + 1}/{F.ROUNDS} done", flush=True)
    RESULTS.write_text(json.dumps({"git_head": F.git("rev-parse", "HEAD").strip(), "ncu": runs}, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(F.REPO)}")


def analyze():
    now = json.loads(RESULTS.read_text())["ncu"]
    base = json.loads(BASELINE.read_text())["ncu"]
    ok = True
    print(f"  {'pair':>10} {'cu_q before':>12} {'cu_q now':>9} {'grouped':>8} {'now/grouped':>12}")
    for p in F.PAIRS:
        before = F.row(base[F.key_of(("cu_q", p))])["total"]
        after = F.row(now[F.key_of(("cu_q", p))])["total"]
        grouped = F.row(base[F.key_of(("cu_q_grouped", p))])["total"]
        r = after / grouped
        ok &= abs(r - 1) <= TOL
        print(f"  {str(p):>10} {before:12.2f} {after:9.2f} {grouped:8.2f} {r:12.3f}")
    bit = all(r["bit_exact"] for v in now.values() for r in v)
    print(f"\n  Q4 (within {TOL:.0%} of cu_q_grouped at every pair): {'PASS' if ok else 'FAIL'};  bit-exact: {bit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "analyze"])
    measure() if ap.parse_args().mode == "run" else analyze()
