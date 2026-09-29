#!/usr/bin/env python3
"""ADR-0029 P2 follow-up, as docs/prereg/ADR-0029.p2-followup-protocol.md fixed it.

    p2_followup.py run      measure, write docs/prereg/ADR-0029.p2-followup-results.json
    p2_followup.py analyze  print the verdicts of F1-F3

Counters only; no timing. `run` refuses a dirty tree.
"""

import argparse
import csv
import io
import json
import pathlib
import random
import re
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BINARY = REPO / "target/release/lyth.exe"
MACHINE = REPO / "fixtures/machine/sm_120.json"
RESULTS = REPO / "docs/prereg/ADR-0029.p2-followup-results.json"

QUADS, N_SPLIT = 1 << 21, 1 << 23
PAIRS = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 8), (8, 0), (4, 1), (1, 4), (22, 2), (8, 4)]
KERNELS = ["cu_q", "cu_q_grouped"]
UNITARY = ["ar=0.6", "ai=0.1", "br=0.2", "bi=0.7", "cr=0.3", "ci=0.5", "dr=0.6", "di=0.4"]
ROUNDS, SEED, TOL = 3, 290, 0.05
METRICS = [
    "lts__t_bytes.sum", "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
]


def widths(c, t):
    return 1 << c, (1 << t) if t < c else (1 << (t - 1))


def key_of(cfg):
    return "control" if cfg is None else f"{cfg[0]}:c={cfg[1][0]},t={cfg[1][1]}"


def command(cfg):
    sets = [a for u in UNITARY for a in ("--set", u)]
    if cfg is None:
        return [str(BINARY), "run", str(REPO / "examples/cu.lyth"), "--machine", str(MACHINE), "-n", str(QUADS), *sets]
    kernel, (c, t) = cfg
    wc, wt = widths(c, t)
    return [str(BINARY), "run", str(REPO / f"examples/{kernel}.lyth"), "--machine", str(MACHINE),
            "-n", str(N_SPLIT), "--set", f"wc={wc}", "--set", f"wt={wt}", *sets]


def find_ncu():
    for base in sorted(pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"), reverse=True):
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def ncu_once(ncu, cfg):
    done = subprocess.run([str(ncu), "--metrics", ",".join(METRICS), "--csv", *command(cfg)], capture_output=True, text=True)
    marker = '"ID","Process ID"'
    if marker not in done.stdout:
        sys.exit(f"{key_of(cfg)}: no profile\n{done.stdout}\n{done.stderr}")
    out = {}
    for r in csv.DictReader(io.StringIO(done.stdout[done.stdout.index(marker):])):
        if r.get("Metric Name") in METRICS:
            out[r["Metric Name"]] = int(re.sub(r"[^0-9]", "", r["Metric Value"]))
    if set(out) != set(METRICS):
        sys.exit(f"{key_of(cfg)}: counters missing")
    exact = re.search(r"exact\s+([0-9.]+) byte per quad", done.stdout)
    return {"counters": out, "bit_exact": "BIT-EXACT" in done.stdout,
            "derived_isolated": float(exact.group(1)) if exact else None}


def measure():
    if git("status", "--porcelain").strip():
        sys.exit("the git tree is dirty")
    ncu = find_ncu()
    rng = random.Random(SEED)
    configs = [None] + [(k, p) for k in KERNELS for p in PAIRS]
    runs = {key_of(c): [] for c in configs}
    for rnd in range(ROUNDS):
        order = configs[:]
        rng.shuffle(order)
        for c in order:
            runs[key_of(c)].append(ncu_once(ncu, c))
        print(f"round {rnd + 1}/{ROUNDS} done", flush=True)
    RESULTS.write_text(json.dumps({"protocol": "docs/prereg/ADR-0029.p2-followup-protocol.md",
                                   "git_head": git("rev-parse", "HEAD").strip(), "quads": QUADS,
                                   "rounds": ROUNDS, "seed": SEED, "ncu": runs}, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(REPO)}")


def row(runs):
    m = {k: statistics.median(r["counters"][k] for r in runs) for k in METRICS}
    ld = m["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]
    return {
        "total": m["lts__t_bytes.sum"] / QUADS,
        "read": m["lts__t_sectors_op_read.sum"] * 32 / QUADS,
        "write": m["lts__t_sectors_op_write.sum"] * 32 / QUADS,
        "l1_hit": m["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"] / ld if ld else 0.0,
        "st_sectors": m["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"] * 32 / QUADS,
        "derived": runs[0]["derived_isolated"],
        "bit_exact": all(r["bit_exact"] for r in runs),
    }


def analyze(path):
    rec = json.loads(path.read_text())
    s = {k: row(v) for k, v in rec["ncu"].items()}
    c = s["control"]
    print(f"commit {rec['git_head'][:12]}; bytes per quad at L2 unless noted\n")
    print(f"  {'':>20} {'total':>7} {'read':>7} {'write':>7} {'L1 hit':>7} {'st@L1':>7} {'derived':>8}")
    print(f"  {'control':>20} {c['total']:7.2f} {c['read']:7.2f} {c['write']:7.2f} {c['l1_hit']:7.1%} {c['st_sectors']:7.1f}")
    for k in KERNELS:
        for p in PAIRS:
            r = s[key_of((k, p))]
            print(f"  {key_of((k, p)):>20} {r['total']:7.2f} {r['read']:7.2f} {r['write']:7.2f} {r['l1_hit']:7.1%} "
                  f"{r['st_sectors']:7.1f} {r['derived']:8.1f}")
    f1_read, f1_write = True, True
    for p in PAIRS:
        r = s[key_of(("cu_q", p))]
        if abs(r["read"] / c["read"] - 1) > TOL:
            f1_read = False
        excess = r["total"] - c["total"]
        if excess > TOL * c["total"] and (r["write"] - c["write"]) / excess < 0.9:
            f1_write = False
    narrow = [p for p in PAIRS if min(p) < 3]
    f2 = all(s[key_of(("cu_q", p))]["l1_hit"] >= 0.5 for p in narrow) and s[key_of(("cu_q", (8, 4)))]["l1_hit"] <= 0.10
    f3 = all(abs(s[key_of(("cu_q_grouped", p))]["total"] / s[key_of(("cu_q", p))]["total"] - 1) <= TOL for p in PAIRS)
    bit = all(v["bit_exact"] for v in s.values())
    print()
    for name, ok in [("F1 reads equal the control's", f1_read), ("F1 writes carry >= 90% of the excess", f1_write),
                     ("F2 L1 serves the partner on reads", f2), ("F3 declaration order does not move the bus", f3),
                     ("every run bit-exact", bit)]:
        print(f"  {name:<44} {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS)
    a = ap.parse_args()
    measure() if a.mode == "run" else analyze(a.results)


if __name__ == "__main__":
    main()
