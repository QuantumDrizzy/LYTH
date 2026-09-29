#!/usr/bin/env python3
"""ADR-0030 step 5: Q2 and Q3, as docs/prereg/ADR-0030.step5-protocol.md fixed them.

    fusion_sweep.py run      measure, write docs/prereg/ADR-0030.step5-results.json
    fusion_sweep.py analyze  print the verdicts

`run` refuses a dirty tree and needs `cargo build --release -p lyth-circuit --examples` first.
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
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
LYTH = REPO / "target/release/lyth.exe"
EX = REPO / "target/release/examples"
MACHINE = REPO / "fixtures/machine/sm_120.json"
RESULTS = REPO / "docs/prereg/ADR-0030.step5-results.json"

N = 1 << 23
SETS = {"A": "7,6,5,4,3", "B": "4,3,2,1,0"}
GATES = [1, 2, 5, 10, 20, 40]
UNITARY = ["ar=0.6", "ai=0.1", "br=0.2", "bi=0.7", "cr=0.3", "ci=0.5", "dr=0.6", "di=0.4"]
METRICS = ["lts__t_bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum"]
MODES = ["unfused", "1", "2", "3", "4", "5"]
Q3 = {"qubits": 23, "depth": 10, "seed": 20260930, "reps": 10}
ROUNDS_Q2, ROUNDS_Q3, TOL = 3, 7, 0.05


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout


def find_ncu():
    for base in sorted(pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"), reverse=True):
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found")


def ncu_once(ncu, cmd):
    done = subprocess.run([str(ncu), "--metrics", ",".join(METRICS), "--csv", *cmd], capture_output=True, text=True)
    marker = '"ID","Process ID"'
    if marker not in done.stdout:
        sys.exit(f"no profile: {' '.join(cmd)}\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    out = {}
    for r in csv.DictReader(io.StringIO(done.stdout[done.stdout.index(marker):])):
        if r.get("Metric Name") in METRICS:
            out[r["Metric Name"]] = int(re.sub(r"[^0-9]", "", r["Metric Value"]))
    derived = re.search(r"derived\s+([0-9.]+) flop/byte", done.stdout)
    return {"counters": out, "bit_exact": "BIT-EXACT" in done.stdout,
            "intensity": float(derived.group(1)) if derived else None}


def measure():
    if git("status", "--porcelain").strip():
        sys.exit("the git tree is dirty")
    ncu = find_ncu()
    work = pathlib.Path(tempfile.mkdtemp())
    configs = {"single": [str(LYTH), "run", str(REPO / "examples/gate_q.lyth"), "--machine", str(MACHINE),
                          "-n", str(N), "--set", f"w={1 << 22}", *[x for u in UNITARY for x in ("--set", u)]]}
    for s, qs in SETS.items():
        for g in GATES:
            path = work / f"{s}{g}.lyth"
            sets = subprocess.run([str(EX / "q2_group.exe"), qs, str(g), "30", str(path)],
                                  capture_output=True, text=True, check=True).stdout.split()
            configs[f"{s}:{g}"] = [str(LYTH), "run", str(path), "--machine", str(MACHINE), "-n", str(N),
                                   *[x for w in sets for x in ("--set", w)]]
    rng = random.Random(305)
    q2 = {k: [] for k in configs}
    for rnd in range(ROUNDS_Q2):
        order = list(configs)
        rng.shuffle(order)
        for k in order:
            q2[k].append(ncu_once(ncu, configs[k]))
        print(f"Q2 round {rnd + 1}/{ROUNDS_Q2} done", flush=True)
    rng = random.Random(306)
    q3 = {m: [] for m in MODES}
    for rnd in range(ROUNDS_Q3):
        order = MODES[:]
        rng.shuffle(order)
        for m in order:
            out = work / f"t_{m}.json"
            subprocess.run([str(EX / "circuit_time.exe"), m, str(Q3["qubits"]), str(Q3["depth"]), str(Q3["seed"]),
                            str(Q3["reps"]), str(out)], check=True)
            q3[m].append(json.loads(out.read_text()))
        print(f"Q3 round {rnd + 1}/{ROUNDS_Q3} done", flush=True)
    RESULTS.write_text(json.dumps({"git_head": git("rev-parse", "HEAD").strip(), "n": N, "q2": q2, "q3": q3}, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(REPO)}")


def analyze():
    rec = json.loads(RESULTS.read_text())
    per = lambda runs, m: statistics.median(r["counters"][m] for r in runs) / N  # noqa: E731
    row = lambda runs: {"dram": per(runs, METRICS[1]) + per(runs, METRICS[2]), "l2": per(runs, METRICS[0]),  # noqa: E731
                        "intensity": runs[0]["intensity"], "exact": all(r["bit_exact"] for r in runs)}
    q2 = {k: row(v) for k, v in rec["q2"].items()}
    s = q2["single"]
    print(f"commit {rec['git_head'][:12]}\n\nQ2, bytes per amplitude (single-gate pass: DRAM {s['dram']:.3f}, L2 {s['l2']:.3f})")
    print(f"  {'group':>7} {'DRAM':>7} {'/single':>8} {'L2':>7} {'/single':>8} {'flop/B':>7}  exact")
    dram_ok, l2_ok = True, True
    for k, r in q2.items():
        if k == "single":
            continue
        rd, rl = r["dram"] / s["dram"], r["l2"] / s["l2"]
        dram_ok &= abs(rd - 1) <= TOL
        if k.startswith("A:"):
            l2_ok &= rl <= 1.10
        print(f"  {k:>7} {r['dram']:7.3f} {rd:8.3f} {r['l2']:7.3f} {rl:8.3f} {r['intensity']:7.2f}  {r['exact']}")
    q3 = {}
    for m, runs in rec["q3"].items():
        meds = [r["ms_median"] for r in runs]
        q3[m] = {"ms": statistics.median(meds), "spread": (max(meds) - min(meds)) / statistics.median(meds),
                 "passes": runs[0]["passes"], "gates": runs[0]["gates"]}
    u = q3["unfused"]
    G = u["gates"]
    print(f"\nQ3, random(23, depth 10), G = {G} gates; unfused {u['ms']:.2f} ms (spread {u['spread']:.1%})")
    print(f"  {'k':>3} {'F':>5} {'F/G':>6} {'ms':>8} {'T/Tu':>6} {'1.15F/G':>8} {'GB/s':>7} {'spread':>7}")
    t_ok = True
    for m in MODES[1:]:
        r = q3[m]
        fg, tt = r["passes"] / G, r["ms"] / u["ms"]
        t_ok &= tt <= 1.15 * fg
        gbs = 16 * N * r["passes"] / (r["ms"] * 1e-3) / 1e9
        print(f"  {m:>3} {r['passes']:5d} {fg:6.3f} {r['ms']:8.2f} {tt:6.3f} {1.15 * fg:8.3f} {gbs:7.1f} {r['spread']:6.1%}")
    ugbs = 16 * N * u["passes"] / (u["ms"] * 1e-3) / 1e9
    print(f"  unfused achieved {ugbs:.1f} GB/s over {u['passes']} passes\n")
    for name, ok in [("Q2 DRAM within 5% of one pass (A and B)", dram_ok), ("Q2 L2 <= 1.10x (A)", l2_ok),
                     ("Q3 T_k/T_u <= 1.15 F_k/G at every k", t_ok),
                     ("every Q2 run bit-exact", all(r["exact"] for r in q2.values()))]:
        print(f"  {name:<44} {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "analyze"])
    measure() if ap.parse_args().mode == "run" else analyze()
