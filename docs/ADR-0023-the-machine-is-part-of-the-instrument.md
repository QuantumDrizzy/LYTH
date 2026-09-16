# ADR-0023 — The machine is part of the instrument

**Status:** Accepted — the harness change is built; two decisions are the operator's
**Date:** 2026-09-16
**Depends on:** ADR-0004 (the probe that measured a host round trip), ADR-0022 (what was running)

On 2026-09-16, during ADR-0022 step 4, the benchmark run ended like this: the card's fans went
to full, the display lost signal, the machine stayed alive — music still playing — and the power
button would not shut it down. It came back only after cutting mains power. `nvidia-smi` on the
way out said:

```
Unable to determine the device handle for GPU0: 0000:03:00.0: GPU is lost.
Reboot the system to recover this GPU
```

This ADR is what the event log said, what it did **not** say, and what changes.

## What actually happened

| | |
|---|---|
| **16 display-driver timeouts today** (`Display` 4101) | 11:12:50 → 15:21:33, each logged "stopped responding and recovered successfully" |
| **1 corrected PCIe error** (`WHEA-Logger` 17) | 15:30:26, on the Intel root port `VEN_8086&DEV_6F04` with the GPU as the secondary device |
| unexpected shutdown (`EventLog` 6008, `Kernel-Power` 41) | 15:36 — the power cut |
| `nvlddmkm` errors 13 / 153 | 1,116 and 26, spanning the same window |

The TDR history is the finding:

| date | driver resets |
|---|---|
| 2026-09-05 | 1 |
| 2026-09-14 | 1 |
| **2026-09-15** | **19** |
| **2026-09-16** | **16** |

**35 of the 37 display-driver resets in the entire log came from the two days this project spent
benchmarking**, for ADR-0020 through ADR-0022. Before that the machine had one in a fortnight.

`TdrDelay` is **not set**, so Windows uses its 2-second default, and the RTX 5060 Ti is the
**only display adapter** — it drives the desktop at 1920x1080 while running every kernel this
project measures. A GPU held by a compute workload on a WDDM display adapter is a GPU the
watchdog is timing.

### One thing that looked damning and is not

The log holds **10,918** corrected PCIe errors. It would be easy to write that as an ongoing
hardware fault, and it is not one:

| date | corrected PCIe errors |
|---|---|
| 2026-09-04 | 10,917 |
| 2026-09-16 | **1** |

A single burst twelve days ago, then nothing until the one at 15:30:26 today — the moment the
card stopped answering. **Zero since the reboot.** The PCIe link is not continuously degrading,
and saying so would have been a scary sentence built out of one old day.

The link runs at **gen 3 x8**. That is also not a fault: this card is x8 by design, and the
root port is a Broadwell-EP, so Gen 3 is the host's ceiling.

## What this project got wrong

Not the kernels. **The harness.**

Sixteen driver resets happened today and **not one of them reached a benchmark's output.** A run
that spans a TDR keeps timing, keeps printing, and produces numbers shaped exactly like numbers.
Several of today's figures were taken in windows that contained one, and there is no way now to
say which.

That is the same failure this project already legislated against in every other dimension:

> `contraction_traffic.py` refuses to report traffic from a kernel that is not bit-exact,
> because a number about the wrong computation is a number about nothing.

A number measured across a driver reset is the same kind of nothing, and nothing was watching
for it.

## What changes here

`bench/gpu_health.py`. A `Watch` reads the TDR count before and after a measurement, and a run
that crossed a reset says so in its own output and **exits non-zero**:

```
  driver   *** 2 DISPLAY-DRIVER RESET(S) DURING THIS RUN ***
           Every number above spans a driver reset and should not be quoted.
```

Wired into `bench/vs_handwritten_matmul.py` and `tools/shared_probe.py`. It is read-only: it
changes no system setting, because the two settings that matter are not the compiler's to
change.

## What is the operator's to decide

**1. `TdrDelay`.** The documented fix for CUDA on a WDDM display GPU is to give the watchdog
more than two seconds:

```
HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
  TdrDelay   REG_DWORD   10        (seconds; reboot to apply)
```

Reversible by deleting the value. The trade is real and should be stated: a genuinely hung
kernel then freezes the desktop for ten seconds instead of two before Windows resets it.

**2. Which mechanism to remove.** Three candidates, and this ADR does **not** claim to have
separated them:

* **`ncu` kernel replay.** The profiler re-runs each kernel many times with cache flushes
  between passes and holds the device across them. Over an hour of `ncu` ran today.
* **The deliberate fault tests.** `--shared-bytes 0` exists to prove a staged kernel really
  stages (ADR-0017), and it works by causing `ILLEGAL_ADDRESS` on purpose. Those run on every
  `cargo test`, and an illegal access on a WDDM display adapter is a textbook TDR trigger. The
  resets cluster in pairs a minute apart, which is what a test binary running several GPU tests
  in sequence looks like.
* **Sustained back-to-back launches.** Least likely on the arithmetic: the longest timed region
  here is ten launches of ~14 ms.

The second one is **ours**, and it is the cheapest to test: run that single test with the GPU
otherwise idle and see whether a 4101 appears. It costs one deliberate driver fault to find out,
on a machine that just needed its power cut, so it is asked rather than assumed.

## What this does not say

It does not say the hardware is failing. One corrected PCIe error on the day it hung, and none
since, is not a diagnosis — it is the link reporting that something went wrong at the moment
something went wrong. If the resets stop when the mechanism above is removed and the card never
drops again, there was nothing to fix. If it drops again with a clean TDR log, that is when the
slot, the riser and the PSU become the question.

And it does not retract any measurement. The figures in ADR-0020 to ADR-0022 have controls of
their own — bit-exactness against the host, bit-identity against nvcc, doubling-scaling, counter
cross-checks — and those do not care about driver resets. What they lack is this one, and from
here they have it.
