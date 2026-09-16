#!/usr/bin/env python3
"""LYTH against a hand-written CUDA C++ kernel of the same schedule.

    python bench/vs_handwritten.py --rounds 7

`Nine Kernels, Measured` listed this as the first thing it did not have, and it is the one
comparison that says something about the *compiler* rather than about the memory system. Both
sides here move the same bytes by construction, so any difference is code generation.

**Fairness is the hard part, not speed.** A hand-written kernel that wins by using a different
schedule has compared two schedules. `bench/cuda/handwritten.cu` is written to the schedule
LYTH emits -- grid-stride, `fmaf`, the same bounds test -- and carries no `__restrict__`,
because LYTH's IR has no aliasing annotation to emit and granting nvrtc one would hand it an
optimisation this is not about.

Both PTX modules are loaded through the same `ctypes` path and launched with the same
`cuLaunchKernel` at the same grid, so the only thing that differs is the instructions.

**Correctness first.** Every kernel is checked against torch before it is timed. A throughput
number from a kernel that computes the wrong thing is a number about nothing -- this project
has produced three of those and each one looked plausible.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import pathlib
import statistics as st
import subprocess
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "gen"))
sys.path.insert(0, str(HERE))

from nvrtc import compile_ptx  # noqa: E402


def driver() -> ctypes.CDLL:
    return ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")


class Module:
    """A PTX module loaded into torch's context, launched the way a binding launches."""

    def __init__(self, cuda: ctypes.CDLL, ptx: str):
        self.cuda = cuda
        self.mod = ctypes.c_void_p()
        self._check("cuModuleLoadData", cuda.cuModuleLoadData(ctypes.byref(self.mod), ptx.encode()))

    def _check(self, what: str, code: int) -> None:
        if code != 0:
            msg = ctypes.c_char_p()
            self.cuda.cuGetErrorString(code, ctypes.byref(msg))
            raise RuntimeError(f"{what}: {msg.value.decode() if msg.value else code}")

    def fn(self, name: str) -> ctypes.c_void_p:
        f = ctypes.c_void_p()
        self._check("cuModuleGetFunction", self.cuda.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()))
        return f

    def launcher(self, name: str, values, grid: int, block: int, shared: int = 0):
        """A callable that launches this kernel. **It owns its arguments.**

        `args` is an array of raw addresses taken from the ctypes objects in `values`, and
        nothing else refers to those objects once this function returns. Without the `keep`
        below Python frees them, `cuLaunchKernel` reads whatever is at those addresses now,
        and the kernel runs with a garbage `n`.

        That is not hypothetical: it reported **3924 GB/s** on a 448 GB/s card. The
        correctness check passed anyway, because it built a launcher and called it in one
        expression -- so `values` was still alive on the stack for that call and dead for
        every timed one. A check that does not cover the path being timed is not a check.
        """
        f = self.fn(name)
        args = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values]
        )
        cuda = self.cuda
        keep = (values, args, f)

        def go(_keep=keep):
            rc = cuda.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, shared, None, args, None)
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernel: {rc}")

        return go


def timed(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / 1e3


def ptx_loop_instructions(ptx: str, entry: str) -> tuple[int, int]:
    """Instructions and branches in the hottest basic block of one entry.

    A crude count on purpose: it reads the emitted PTX rather than modelling it, and it is
    reported beside a SASS measurement precisely because ADR-0014 says an instruction count in
    PTX is a statement about code that ptxas is free to rewrite.
    """
    body, inside = [], False
    for line in ptx.splitlines():
        s = line.strip()
        if s.startswith(".visible .entry"):
            inside = entry in s
            continue
        if inside and s == "}":
            break
        if inside:
            body.append(s)
    # the loop is between the last backward label and the branch that returns to it
    labels = [i for i, s in enumerate(body) if s.endswith(":") and "BB" in s or s.startswith("$L_loop")]
    if not labels:
        return (0, 0)
    start = labels[0]
    end = max(i for i, s in enumerate(body) if s.endswith(";") and "bra" in s)
    block = [s for s in body[start + 1 : end + 1] if s and not s.startswith("//") and not s.endswith(":")]
    return (len(block), sum(1 for s in block if "bra" in s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1 << 26)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--binary", type=pathlib.Path, default=REPO / "target/release/lyth.exe")
    ap.add_argument("--machine", type=pathlib.Path, default=REPO / "fixtures/machine/sm_120.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("skipped: no CUDA device")
        return 77

    # --- LYTH's saxpy, through its own generated binding ---------------------------
    gen = HERE / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    if not (gen / "saxpy.py").exists():
        if not args.binary.exists():
            sys.exit(f"{args.binary} is not built. Run `cargo build --release`.")
        subprocess.run(
            [str(args.binary), "build", str(REPO / "examples/saxpy.lyth"),
             "--machine", str(args.machine),
             "-o", "nul" if sys.platform == "win32" else "/dev/null",
             "--bind-py", str(gen / "saxpy.py")],
            check=True, capture_output=True,
        )
    lyth = importlib.import_module("saxpy")

    # --- the hand-written one, compiled here ---------------------------------------
    cu = (HERE / "cuda" / "handwritten.cu").read_text(encoding="utf-8")
    hand_ptx = compile_ptx(cu, "handwritten.cu")

    n = args.n
    torch.manual_seed(3)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y0 = torch.randn(n, device="cuda", dtype=torch.float32)
    a = 2.0

    cuda = driver()
    hand = Module(cuda, hand_ptx)

    grid, block = lyth.grid(n), lyth.BLOCK
    print(f"\n  n = {n:,}   grid {grid:,} x block {block}   (LYTH's launch, used for both)\n")

    # --- correctness, before anything is timed --------------------------------------
    want = a * x + y0
    results = {}
    # The launchers checked here are the launchers timed below -- built once, used twice.
    # Building a second one for the check is how the argument-lifetime bug hid.
    yl, yh = y0.clone(), y0.clone()
    k_lyth = lyth.Kernel()
    checked = {
        "LYTH": (yl, lambda: k_lyth.launch(n, a, x.data_ptr(), yl.data_ptr())),
        "nvrtc": (yh, hand.launcher(
            "saxpy_gridstride",
            [ctypes.c_uint32(n), ctypes.c_float(a), ctypes.c_uint64(x.data_ptr()),
             ctypes.c_uint64(yh.data_ptr())],
            grid, block)),
    }
    for label, (y, run) in checked.items():
        run()
        torch.cuda.synchronize()
        exact = torch.equal(y, want)
        results[label] = y.clone()
        print(f"  {label:<6} {'bit-exact against torch' if exact else '*** DIFFERS ***'}")
        if not exact:
            sys.exit(f"{label} computes the wrong thing; a timing would mean nothing")
        y.copy_(y0)
    if not torch.equal(results["LYTH"], results["nvrtc"]):
        sys.exit("the two kernels disagree with each other")
    print("  and with each other\n")

    # --- what each compiler emitted -------------------------------------------------
    lyth_i, lyth_b = ptx_loop_instructions(lyth.PTX, "saxpy")
    hand_i, hand_b = ptx_loop_instructions(hand_ptx, "saxpy_gridstride")
    print(f"  {'':<6} {'PTX loop body':>15} {'branches':>10}")
    print(f"  {'LYTH':<6} {lyth_i:>15} {lyth_b:>10}")
    print(f"  {'nvrtc':<6} {hand_i:>15} {hand_b:>10}")
    print("  (PTX, before ptxas. ADR-0014: bytes survive ptxas and instructions do not.)\n")

    # --- interleaved, same grid, same block -----------------------------------------
    calls = {
        "LYTH": checked["LYTH"][1],
        "nvrtc": checked["nvrtc"][1],
        "torch": lambda: yh.add_(x, alpha=a),
    }
    rows = {k: [] for k in calls}
    for r in range(args.rounds):
        for label, fn in calls.items():
            rows[label].append(12 * n / timed(fn, args.reps) / 1e9)
        print(f"  round {r + 1}/{args.rounds}", file=sys.stderr)

    print(f"  {'':<6} {'GB/s median':>12} {'min':>8} {'max':>8}")
    med = {}
    for label in calls:
        v = sorted(rows[label])
        med[label] = st.median(v)
        print(f"  {label:<6} {med[label]:>12.1f} {v[0]:>8.1f} {v[-1]:>8.1f}")
    lo = min(rows["LYTH"]) / max(rows["nvrtc"])
    hi = max(rows["LYTH"]) / min(rows["nvrtc"])
    print(f"\n  LYTH / nvrtc = {med['LYTH'] / med['nvrtc']:.1%}  [{lo:.0%}-{hi:.0%}]")
    print(f"  {args.rounds} rounds x {args.reps} launches, interleaved, same grid and block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
