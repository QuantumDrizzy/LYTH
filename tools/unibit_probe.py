#!/usr/bin/env python3
"""What the Unibit machine actually does per cycle. ADR-0025 step 1.

    python tools/unibit_probe.py            # -> fixtures/machine/unibit.json

`fixtures/machine/sm_120.json` opens with

    EVERY NUMBER HERE IS MEASURED ON THIS MACHINE, NOT A DATASHEET

and this is that rule applied to the second machine. Every figure below comes out of the
emulator's own counters, from programs written here and run here.

**The unit is a cycle, and that is the finding.** Unibit has no clock: `src/cpu.rs` charges one
cycle per instruction, three more for a division, three more for a mispredicted branch, and
stops. There is no frequency anywhere in the machine, so **`bandwidth_gbs` is meaningless for
it** -- the same shape of mistake ADR-0022 caught when it found the shared pipe wanted accesses
per second rather than bytes per second. A machine file that invented a megahertz to make the
existing field fit would be a datasheet with extra steps.

So the Unibit machine declares `bytes_per_cycle` and `flops_per_cycle`, and its ceiling is in
cycles. That is not a degradation: on a machine whose only clock is the instruction count, a
cycle is the most honest unit there is.

Two probes, each unrolled 16 times so the loop's own `addi`/`bne` is a small fraction of the
measurement rather than a doubling of it -- the batching argument from exercise 06, reached by
the same route on a very different machine:

* **memory**: `LQ` in a tight unrolled loop. 32 bytes per instruction, one instruction per
  cycle, and the counter says whether that is really what happened.
* **arithmetic**: `VFMA` in the same shape. Eight f32 lanes, two flops each.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
UNIBIT = REPO.parent / "Unibit"

UNROLL = 16
ITERS = 512


def memory_probe() -> str:
    """`LQ` as fast as the machine will take them: 32 bytes per instruction."""
    body = "\n".join(f"        lq      t{i % 3}, {(i % 4) * 32}(s0)" for i in range(UNROLL))
    return f"""; ADR-0025 step 1: how many bytes a load moves per cycle.
        .data
buf:    .word 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        .word 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32

        .text
        .global _start
_start:
        la      s0, buf
        li      s1, {ITERS}
loop:
{body}
        addi    s1, s1, -1
        bne     s1, zero, loop
        halt
"""


def flop_probe() -> str:
    """`VFMA` as fast as the machine will take them: eight lanes, two flops each."""
    body = "\n".join(f"        vfma    t{i % 3}, t3, t4" for i in range(UNROLL))
    return f"""; ADR-0025 step 1: how many flops an instruction retires per cycle.
        .data
ones:   .word 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000

        .text
        .global _start
_start:
        la      s0, ones
        lq      t3, 0(s0)
        lq      t4, 0(s0)
        li      s1, {ITERS}
loop:
{body}
        addi    s1, s1, -1
        bne     s1, zero, loop
        halt
"""


def run(src: str, tmp: pathlib.Path, name: str) -> dict:
    f = tmp / f"{name}.uasm"
    f.write_text(src, encoding="utf-8")
    out = subprocess.run(
        ["cargo", "run", "--quiet", "--", "run", str(f)],
        cwd=UNIBIT, capture_output=True, text=True, timeout=600,
        # The report is drawn with box characters and the banner is ASCII art, so the host's
        # cp1252 default cannot decode it. Replacing rather than failing: the counters this
        # reads are plain digits and a mangled banner costs nothing.
        encoding="utf-8", errors="replace",
    )
    text = out.stdout + out.stderr
    got = {}
    for key, pat in (
        ("instructions", r"Total Instructions Retired:\s*(\d+)"),
        ("cycles", r"Cycles:\s*(\d+)"),
        ("float_ops", r"Packed f32 \(8 lanes\):\s*(\d+)"),
        ("reads", r"Memory Reads/Writes:\s*(\d+)"),
    ):
        m = re.search(pat, text)
        if m:
            got[key] = int(m.group(1))
    if "cycles" not in got:
        sys.exit(f"{name}: could not read the counters\n{text[-800:]}")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--emit", type=pathlib.Path,
                    default=REPO / "fixtures/machine/unibit.json")
    args = ap.parse_args()

    if not (UNIBIT / "Cargo.toml").exists():
        print(f"skipped: no Unibit at {UNIBIT}")
        return 77

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        mem = run(memory_probe(), tmp, "mem")
        flop = run(flop_probe(), tmp, "flop")

    # The loop's own `addi` and `bne` are in `instructions` too, which is why the rate is
    # computed from the counted loads rather than from the instruction total. The unroll is
    # what keeps the difference small; reporting it is what keeps the claim honest.
    load_bytes = mem["reads"] * 32
    bytes_per_cycle = load_bytes / mem["cycles"]
    ideal_bpc = UNROLL * 32 / (UNROLL + 2)

    flops = flop["float_ops"] * 16  # eight lanes, a multiply and an add each
    flops_per_cycle = flops / flop["cycles"]
    ideal_fpc = UNROLL * 16 / (UNROLL + 2)

    print(f"  memory  {mem['reads']:>6} loads, {mem['cycles']:>6} cycles"
          f"  -> {bytes_per_cycle:6.2f} bytes/cycle   (loop-free limit {32}, "
          f"this loop's {ideal_bpc:.2f})")
    print(f"  flops   {flop['float_ops']:>6} vfma,  {flop['cycles']:>6} cycles"
          f"  -> {flops_per_cycle:6.2f} flops/cycle   (loop-free limit {16}, "
          f"this loop's {ideal_fpc:.2f})")

    ridge = flops_per_cycle / bytes_per_cycle
    print(f"\n  ridge   {ridge:.4f} flop/byte")
    print(f"          sm_120's is 36.9. A kernel that is memory-bound on the GPU can be "
          f"compute-bound here,")
    print(f"          which is the whole reason a second machine is worth having.")

    machine = {
        "schema": "lyth-machine/0.1",
        "id": "unibit",
        "time_unit": "cycle",
        "notes": [
            "EVERY NUMBER HERE IS MEASURED ON THIS MACHINE, NOT A DATASHEET. Produced by "
            "tools/unibit_probe.py against the emulator's own counters.",
            "This machine has no clock. src/cpu.rs charges one cycle per instruction, three "
            "more for a division and three for a mispredicted branch, and there is no "
            "frequency anywhere in it. So rates are per CYCLE and `bandwidth_gbs` is left at "
            "0.0 -- inventing a megahertz to make the existing field fit would be a datasheet "
            "with extra steps. Same shape as ADR-0022 finding that the shared pipe wanted "
            "accesses per second rather than bytes per second.",
            "There is no cache hierarchy and no shared memory, so `mem` and `reg` are the only "
            "levels. A kernel that declares `dram -> smem -> reg` should be REFUSED here rather "
            "than quietly lowered: ADR-0022 made the ceiling a minimum over the levels a "
            "machine names, and a machine that names no shared level cannot stage.",
            "Unibit is an emulator. These are its counters, not silicon.",
        ],
        "levels": [
            {
                "name": "mem",
                "capacity": "flat, sized at boot",
                "bandwidth_gbs": 0.0,
                "bytes_per_cycle": round(bytes_per_cycle, 4),
                "latency_cyc": 1,
            },
            {
                "name": "reg",
                "capacity": "32 x 256-bit",
                "bandwidth_gbs": 0.0,
                "latency_cyc": 0,
            },
        ],
        "flops_per_cycle": round(flops_per_cycle, 4),
        "ridge_flop_per_byte": {
            "fp32": round(ridge, 4),
            "note": "flops_per_cycle / bytes_per_cycle, both measured on this emulator. "
                    "Unlike sm_120's, both terms are exact rather than achieved: an emulator "
                    "that charges one cycle per instruction has no variance to average out.",
        },
        "ops": [
            {"name": "vfma.f32", "status": "present", "at": "reg"},
            {"name": "vfadd.f32", "status": "present", "at": "reg"},
            {"name": "vfmul.f32", "status": "present", "at": "reg"},
            {"name": "vreduce", "status": "present", "at": "reg"},
            {"name": "smem", "status": "absent", "at": "mem"},
            {"name": "fma.f64", "status": "absent", "at": "reg"},
        ],
        "measurement": {
            "method": "tools/unibit_probe.py: LQ and VFMA in loops unrolled "
                      f"{UNROLL}x over {ITERS} iterations, read from the emulator's counters",
            "memory_loads": mem["reads"],
            "memory_cycles": mem["cycles"],
            "vfma_count": flop["float_ops"],
            "vfma_cycles": flop["cycles"],
            "loop_overhead": "2 instructions per unrolled body (addi, bne); the unroll is what "
                             "makes that a small fraction, and the loop-free limits are "
                             "printed beside the measured rate so the gap is visible",
        },
    }
    args.emit.parent.mkdir(parents=True, exist_ok=True)
    args.emit.write_text(json.dumps(machine, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.emit.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
