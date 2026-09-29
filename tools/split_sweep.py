#!/usr/bin/env python3
"""ADR-0028 step 5: the falsification sweep of P1-P4, as docs/prereg/ADR-0028.step5-protocol.md
fixed it before the first launch.

    split_sweep.py run      measure, and write docs/prereg/ADR-0028.step5-results.json
    split_sweep.py analyze  read that file and print the verdict of each prediction

The two are separate so the verdict rules cannot see the measurement loop, and so `analyze` can be
re-run on the recorded numbers by anyone. Every threshold below is the protocol's, not tuned.

`run` refuses to start on a dirty git tree: the results record the commit they were taken at, and
a commit that does not contain the binary's source is a record of nothing.
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
RESULTS = REPO / "docs/prereg/ADR-0028.step5-results.json"

PAIRS = 1 << 22  # what both kernels walk
N_SPLIT = 1 << 23  # amplitudes: four buffers of 2^23 f32
WIDTHS = [1 << q for q in range(23)]  # every qubit of the register
TIME_ROUNDS = 7
NCU_ROUNDS = 3
REPS = 200
SEED = 28
TOL = 0.05
NOISE = 0.01  # counter noise allowed between neighbouring widths (P2 shape)
METRICS = ["lts__t_bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum"]


def key_of(w):
    return "control" if w is None else f"w={w}"


def command(w):
    if w is None:
        return [str(BINARY), "run", str(REPO / "examples/hadamard.lyth"), "--machine", str(MACHINE),
                "-n", str(PAIRS)]
    return [str(BINARY), "run", str(REPO / "examples/hadamard_q.lyth"), "--machine", str(MACHINE),
            "-n", str(N_SPLIT), "--set", f"w={w}"]


def find_ncu():
    for base in sorted(pathlib.Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Compute*"),
                       reverse=True):
        for name in ("ncu.bat", "ncu.exe"):
            if (base / name).exists():
                return base / name
    sys.exit("ncu not found under 'C:/Program Files/NVIDIA Corporation/Nsight Compute*'")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def ncu_once(ncu, w):
    done = subprocess.run([str(ncu), "--metrics", ",".join(METRICS), "--csv", *command(w)],
                          capture_output=True, text=True)
    marker = '"ID","Process ID"'
    if marker not in done.stdout:
        sys.exit(f"{key_of(w)}: no profile\n{done.stdout}\n{done.stderr}")
    out = {}
    for r in csv.DictReader(io.StringIO(done.stdout[done.stdout.index(marker):])):
        if r.get("Metric Name") in METRICS:
            # ncu writes the value in the host locale: 67.123.200 is one number, in bytes.
            out[r["Metric Name"]] = int(re.sub(r"[^0-9]", "", r["Metric Value"]))
    if set(out) != set(METRICS):
        sys.exit(f"{key_of(w)}: counters missing from the profile: {sorted(set(METRICS) - set(out))}")
    return {"counters": out, "bit_exact": "BIT-EXACT" in done.stdout}


def time_once(w):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        path = pathlib.Path(t.name)
    done = subprocess.run(command(w) + ["--time", str(REPS), "--json", str(path)],
                          capture_output=True, text=True)
    bit_exact = "BIT-EXACT" in done.stdout
    median = None
    if bit_exact:
        median = json.loads(path.read_text())["ms_median"]
    path.unlink(missing_ok=True)
    return {"ms_median": median, "bit_exact": bit_exact,
            "verify_failed": "verify FAILED" in done.stderr + done.stdout}


def measure():
    if git("status", "--porcelain").strip():
        sys.exit("the git tree is dirty: commit the tool and the protocol first, so the results "
                 "name the commit they were taken at")
    if not BINARY.exists():
        sys.exit(f"{BINARY} not built (cargo build --release)")
    ncu = find_ncu()
    rng = random.Random(SEED)
    configs = [None] + WIDTHS
    ncu_runs = {key_of(w): [] for w in configs}
    time_runs = {key_of(w): [] for w in configs}

    for rnd in range(NCU_ROUNDS):
        order = configs[:]
        rng.shuffle(order)
        for w in order:
            ncu_runs[key_of(w)].append(ncu_once(ncu, w))
        print(f"ncu round {rnd + 1}/{NCU_ROUNDS} done", flush=True)
    for rnd in range(TIME_ROUNDS):
        order = configs[:]
        rng.shuffle(order)
        for w in order:
            time_runs[key_of(w)].append(time_once(w))
        print(f"timing round {rnd + 1}/{TIME_ROUNDS} done", flush=True)

    record = {
        "protocol": "docs/prereg/ADR-0028.step5-protocol.md",
        "git_head": git("rev-parse", "HEAD").strip(),
        "pairs": PAIRS, "n_split": N_SPLIT, "reps": REPS, "seed": SEED,
        "ncu_rounds": NCU_ROUNDS, "time_rounds": TIME_ROUNDS,
        "ncu": ncu_runs, "time": time_runs,
    }
    RESULTS.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(REPO)}")


# ---------------------------------------------------------------------------------- the verdicts

def summarise(rec):
    """Per configuration: bytes per pair (median of the ncu launches), time (median of the
    process medians) and the run-to-run spread of those."""
    out = {}
    for k in rec["ncu"]:
        runs = rec["ncu"][k]
        row = {m: statistics.median(r["counters"][m] for r in runs) / PAIRS for m in METRICS}
        row["dram"] = row["dram__bytes_op_read.sum"] + row["dram__bytes_op_write.sum"]
        row["lts"] = row["lts__t_bytes.sum"]
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
    ws = WIDTHS
    at = lambda w: s[key_of(w)]  # noqa: E731
    v = {"eps": eps, "control_spread": c["spread"]}

    # P1: the payload does not move. Judged against the control under the same conditions.
    v["P1_invariance"] = all(abs(at(w)["dram"] / c["dram"] - 1) <= TOL for w in ws)
    v["P1_control_dram_per_pair"] = c["dram"]
    v["P1_absolute_confirmed"] = abs(c["dram"] / 32.0 - 1) <= TOL  # the frozen text: 32 B/pair

    # P2: the bus step.
    r = {w: at(w)["lts"] / c["lts"] for w in ws}
    v["P2_ratio_by_w"] = r
    ge8 = all(abs(r[w] - 1) <= TOL for w in ws if w >= 8)
    lt8 = all(r[w] > 1 + TOL for w in (1, 2, 4))
    nonincreasing = all(at(b)["lts"] <= at(a)["lts"] * (1 + NOISE) for a, b in zip(ws, ws[1:]))
    v["P2_shape"] = ge8 and lt8 and nonincreasing
    v["P2_shape_parts"] = {"w>=8 equals the payload": ge8, "w in {1,2,4} exceeds it": lt8,
                           "non-increasing in w": nonincreasing}
    v["P2_magnitude_refined_2x"] = all(1.8 <= r[w] <= 2.2 for w in (1, 2, 4))
    v["P2_bound_frozen_8x"] = all(r[w] <= 8 for w in ws)

    # P3: time.
    T = {w: at(w)["ms"] / c["ms"] for w in ws}
    v["P3_ratio_by_w"] = T
    p3_hi = all(abs(T[w] - 1) <= eps for w in ws if w >= 8)
    p3_lo = all(T[w] <= max(1.0, r[w]) * (1 + eps) for w in (1, 2, 4))
    v["P3"] = p3_hi and p3_lo
    v["P3_parts"] = {"w>=8 equals the control": p3_hi, "w in {1,2,4} within the bus ratio": p3_lo}

    # P4: correctness on the PTX back end at this size.
    v["P4"] = all(at(w)["bit_exact"] for w in ws) and c["bit_exact"]
    return s, v


def analyze(path):
    rec = json.loads(path.read_text())
    s, v = verdicts(rec)
    c = s["control"]
    print(f"commit {rec['git_head'][:12]}, {rec['ncu_rounds']} ncu launches and {rec['time_rounds']} "
          f"timing processes per configuration\n")
    print(f"  {'':>10} {'DRAM B/pair':>12} {'L2 B/pair':>10} {'L2/ctrl':>8} {'ms':>8} {'t/ctrl':>7} "
          f"{'spread':>7}  bit-exact")
    for k in ["control"] + [key_of(w) for w in WIDTHS]:
        row = s[k]
        w = None if k == "control" else int(k[2:])
        r = "" if w is None else f"{v['P2_ratio_by_w'][w]:8.3f}"
        t = "" if w is None else f"{v['P3_ratio_by_w'][w]:7.3f}"
        print(f"  {k:>10} {row['dram']:12.2f} {row['lts']:10.2f} {r:>8} {row['ms']:8.4f} {t:>7} "
              f"{row['spread'] * 100:6.1f}%  {row['bit_exact']}")
    print(f"\n  tol {TOL:.0%}; control's own run-to-run spread {v['control_spread']:.1%}; eps = {v['eps']:.1%}\n")
    print(f"  P1 invariance (DRAM per pair within {TOL:.0%} of the control at every w): "
          f"{'PASS' if v['P1_invariance'] else 'FAIL'}")
    print(f"  P1 as stated, the control at 32 B/pair: "
          f"{'confirmed' if v['P1_absolute_confirmed'] else 'NOT confirmed'} "
          f"(it moves {v['P1_control_dram_per_pair']:.2f} B/pair)")
    print(f"  P2 shape:                {'PASS' if v['P2_shape'] else 'FAIL'}   {v['P2_shape_parts']}")
    print(f"  P2 magnitude, refined 2x: {'PASS' if v['P2_magnitude_refined_2x'] else 'FAIL'}   "
          f"(w=1,2,4: {[round(v['P2_ratio_by_w'][w], 3) for w in (1, 2, 4)]})")
    print(f"  P2 bound, frozen 8x:      {'PASS' if v['P2_bound_frozen_8x'] else 'FAIL'}")
    print(f"  P3 time:                 {'PASS' if v['P3'] else 'FAIL'}   {v['P3_parts']}")
    print(f"  P4 bit-exact on PTX:     {'PASS' if v['P4'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS,
                    help="analyze: the recorded numbers to read (default: the frozen results file)")
    args = ap.parse_args()
    if args.mode == "run":
        measure()
    else:
        analyze(args.results)


if __name__ == "__main__":
    main()
