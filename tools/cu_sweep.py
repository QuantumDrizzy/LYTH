#!/usr/bin/env python3
"""ADR-0029 step 5: the falsification sweep of P1-P3, as docs/prereg/ADR-0029.step5-protocol.md
fixed it before the first launch.

    cu_sweep.py run      measure, and write docs/prereg/ADR-0029.step5-results.json
    cu_sweep.py analyze  read that file and print the verdict of each prediction

Same shape and same instruments as tools/split_sweep.py (ADR-0028), which stays frozen as it was.
`run` refuses a dirty tree.
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
BINARY = REPO / "target/release/lyth.exe"
MACHINE = REPO / "fixtures/machine/sm_120.json"
RESULTS = REPO / "docs/prereg/ADR-0029.step5-results.json"

QUADS = 1 << 21
N_SPLIT = 1 << 23
QUBITS = [0, 1, 2, 3, 4, 8, 16, 22]
PAIRS = [(c, t) for c in QUBITS for t in QUBITS if c != t]
UNITARY = ["ar=0.6", "ai=0.1", "br=0.2", "bi=0.7", "cr=0.3", "ci=0.5", "dr=0.6", "di=0.4"]
TIME_ROUNDS, NCU_ROUNDS, REPS, SEED = 7, 3, 200, 29
TOL, BOUND = 0.05, 1.10
METRICS = ["lts__t_bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum"]


def widths(c, t):
    return 1 << c, (1 << t) if t < c else (1 << (t - 1))


def key_of(ct):
    return "control" if ct is None else f"c={ct[0]},t={ct[1]}"


def command(ct):
    sets = [a for u in UNITARY for a in ("--set", u)]
    if ct is None:
        return [str(BINARY), "run", str(REPO / "examples/cu.lyth"), "--machine", str(MACHINE),
                "-n", str(QUADS), *sets]
    wc, wt = widths(*ct)
    return [str(BINARY), "run", str(REPO / "examples/cu_q.lyth"), "--machine", str(MACHINE),
            "-n", str(N_SPLIT), "--set", f"wc={wc}", "--set", f"wt={wt}", *sets]


def find_ncu():
    for base in sorted(pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"),
                       reverse=True):
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def ncu_once(ncu, ct):
    done = subprocess.run([str(ncu), "--metrics", ",".join(METRICS), "--csv", *command(ct)],
                          capture_output=True, text=True)
    marker = '"ID","Process ID"'
    if marker not in done.stdout:
        sys.exit(f"{key_of(ct)}: no profile\n{done.stdout}\n{done.stderr}")
    out = {}
    for r in csv.DictReader(io.StringIO(done.stdout[done.stdout.index(marker):])):
        if r.get("Metric Name") in METRICS:
            out[r["Metric Name"]] = int(re.sub(r"[^0-9]", "", r["Metric Value"]))
    if set(out) != set(METRICS):
        sys.exit(f"{key_of(ct)}: counters missing")
    exact = re.search(r"exact\s+([0-9.]+) byte per quad", done.stdout)
    return {"counters": out, "bit_exact": "BIT-EXACT" in done.stdout,
            "derived_isolated": float(exact.group(1)) if exact else None}


def time_once(ct):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        path = pathlib.Path(t.name)
    done = subprocess.run(command(ct) + ["--time", str(REPS), "--json", str(path)],
                          capture_output=True, text=True)
    ok = "BIT-EXACT" in done.stdout
    median = json.loads(path.read_text())["ms_median"] if ok else None
    path.unlink(missing_ok=True)
    return {"ms_median": median, "bit_exact": ok}


def measure():
    if git("status", "--porcelain").strip():
        sys.exit("the git tree is dirty: commit the tool and the protocol first")
    ncu = find_ncu()
    rng = random.Random(SEED)
    configs = [None] + PAIRS
    ncu_runs = {key_of(ct): [] for ct in configs}
    time_runs = {key_of(ct): [] for ct in configs}
    for rnd in range(NCU_ROUNDS):
        order = configs[:]
        rng.shuffle(order)
        for ct in order:
            ncu_runs[key_of(ct)].append(ncu_once(ncu, ct))
        print(f"ncu round {rnd + 1}/{NCU_ROUNDS} done", flush=True)
    for rnd in range(TIME_ROUNDS):
        order = configs[:]
        rng.shuffle(order)
        for ct in order:
            time_runs[key_of(ct)].append(time_once(ct))
        print(f"timing round {rnd + 1}/{TIME_ROUNDS} done", flush=True)
    record = {"protocol": "docs/prereg/ADR-0029.step5-protocol.md", "git_head": git("rev-parse", "HEAD").strip(),
              "quads": QUADS, "n_split": N_SPLIT, "reps": REPS, "seed": SEED,
              "ncu_rounds": NCU_ROUNDS, "time_rounds": TIME_ROUNDS, "ncu": ncu_runs, "time": time_runs}
    RESULTS.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(REPO)}")


def summarise(rec):
    out = {}
    for k, runs in rec["ncu"].items():
        row = {m: statistics.median(r["counters"][m] for r in runs) / QUADS for m in METRICS}
        row["dram"] = row["dram__bytes_op_read.sum"] + row["dram__bytes_op_write.sum"]
        row["lts"] = row["lts__t_bytes.sum"]
        row["derived"] = runs[0]["derived_isolated"]
        ts = [r["ms_median"] for r in rec["time"][k] if r["ms_median"] is not None]
        row["ms"] = statistics.median(ts) if ts else None
        row["spread"] = (max(ts) - min(ts)) / statistics.median(ts) if len(ts) > 1 else None
        row["bit_exact"] = all(r["bit_exact"] for r in runs) and all(r["bit_exact"] for r in rec["time"][k])
        out[k] = row
    return out


def verdicts(rec):
    s = summarise(rec)
    c = s["control"]
    eps = max(TOL, c["spread"] or 0.0)
    at = lambda ct: s[key_of(ct)]  # noqa: E731
    r = {ct: at(ct)["lts"] / c["lts"] for ct in PAIRS}
    T = {ct: at(ct)["ms"] / c["ms"] for ct in PAIRS}
    v = {"eps": eps, "control_spread": c["spread"], "r": r, "T": T}
    v["P1"] = all(abs(at(ct)["dram"] / c["dram"] - 1) <= TOL for ct in PAIRS)
    v["P2_equal_where_runs_fill_a_sector"] = all(abs(r[ct] - 1) <= TOL for ct in PAIRS if min(ct) >= 3)
    v["P2_bounded_1.10"] = all(r[ct] <= BOUND for ct in PAIRS)
    v["P2_derived_is_upper_bound"] = all(at(ct)["derived"] >= at(ct)["lts"] / (1 + TOL) for ct in PAIRS)
    v["P3"] = all(abs(T[ct] - 1) <= eps for ct in PAIRS)
    v["P4"] = c["bit_exact"] and all(at(ct)["bit_exact"] for ct in PAIRS)
    return s, v


def analyze(path):
    rec = json.loads(path.read_text())
    s, v = verdicts(rec)
    c = s["control"]
    print(f"commit {rec['git_head'][:12]}, {rec['ncu_rounds']} ncu launches and {rec['time_rounds']} timing processes each\n")
    print(f"  {'':>12} {'DRAM B/q':>9} {'L2 B/q':>8} {'L2/ctl':>7} {'derived':>8} {'ms':>8} {'t/ctl':>6} {'spread':>7}  exact")
    print(f"  {'control':>12} {c['dram']:9.2f} {c['lts']:8.2f} {'':>7} {'':>8} {c['ms']:8.4f} {'':>6} {c['spread'] * 100:6.1f}%  {c['bit_exact']}")
    for ct in PAIRS:
        row = s[key_of(ct)]
        print(f"  {key_of(ct):>12} {row['dram']:9.2f} {row['lts']:8.2f} {v['r'][ct]:7.3f} {row['derived']:8.1f} "
              f"{row['ms']:8.4f} {v['T'][ct]:6.3f} {row['spread'] * 100:6.1f}%  {row['bit_exact']}")
    worst_r = max(v["r"].items(), key=lambda kv: kv[1])
    worst_t = max(v["T"].items(), key=lambda kv: abs(kv[1] - 1))
    print(f"\n  tol {TOL:.0%}; control spread {v['control_spread']:.1%}; eps {v['eps']:.1%}")
    print(f"  largest L2 ratio {worst_r[1]:.3f} at {key_of(worst_r[0])}; largest time deviation {worst_t[1]:.3f} at {key_of(worst_t[0])}\n")
    for name in ["P1", "P2_equal_where_runs_fill_a_sector", "P2_bounded_1.10", "P2_derived_is_upper_bound", "P3", "P4"]:
        print(f"  {name:<36} {'PASS' if v[name] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS)
    args = ap.parse_args()
    if args.mode == "run":
        measure()
    else:
        analyze(args.results)


if __name__ == "__main__":
    main()
