#!/usr/bin/env python3
"""Whether the driver survived the measurement. ADR-0023.

    python bench/gpu_health.py            # the current state, and today's history

On 2026-09-16 a benchmark run ended with the GPU gone from the bus -- `nvidia-smi` reporting
`GPU is lost. Reboot the system to recover this GPU`, fans at full, the display dark, and the
machine only recoverable by cutting mains power. The Windows event log showed what the harness
had not:

* **16 display-driver timeouts (TDR, event 4101) that day**, from 11:12 to 15:21, each one
  logged as "stopped responding and recovered successfully";
* **35 of the 37 TDRs in the whole log came from the two days of benchmarking** for ADR-0020
  through ADR-0022. Before that: one on 5 September, one on 14 September.

Every one of those was invisible to the benchmark that caused it. A run that spans a driver
reset keeps timing, keeps printing, and produces numbers that look exactly like numbers.

**So this module makes the harness say so.** It reads the TDR count before and after, and a
measurement that crossed a reset is reported as one. This is the same rule the rest of the
project already follows for correctness -- `contraction_traffic.py` refuses to report traffic
from a kernel that is not bit-exact, because a number about the wrong computation is a number
about nothing. A number measured across a driver reset is the same kind of nothing.

It is deliberately read-only. Nothing here changes a system setting; `TdrDelay` and the PCIe
link are the operator's to decide about, and ADR-0023 says what they are.
"""

from __future__ import annotations

import subprocess
import sys

# `Display` 4101: "The display driver nvlddmkm stopped responding and has successfully
# recovered." Logged whatever the UI language, which is why this matches on the numeric id and
# provider rather than on the message text.
TDR_QUERY = (
    "(Get-WinEvent -FilterHashtable @{LogName='System';Id=4101;ProviderName='Display'} "
    "-ErrorAction SilentlyContinue | Measure-Object).Count"
)


def tdr_count() -> int | None:
    """How many display-driver resets this machine has logged. `None` off Windows."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", TDR_QUERY],
            capture_output=True, text=True, timeout=60,
        )
        return int(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        # A health check that fails must not fail the benchmark. It reports that it could not
        # look, which is different from reporting that nothing happened.
        return None


def smi(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return None
        return [f.strip() for f in out.stdout.strip().split(",")]
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


class Watch:
    """Bracket a measurement and report what the driver did during it.

        with Watch() as w:
            ...
        w.report()

    `w.clean` is False if the driver reset while the block ran, and a caller that quotes a
    throughput without checking it is quoting a number it has no reason to trust.
    """

    def __init__(self) -> None:
        self.before: int | None = None
        self.after: int | None = None

    def __enter__(self) -> Watch:
        self.before = tdr_count()
        t = smi("temperature.gpu,power.draw,clocks.sm")
        if t:
            print(f"  gpu      {t[0]}C, {float(t[1]):.0f} W, {t[2]} MHz at the start")
        return self

    def __exit__(self, *_) -> None:
        self.after = tdr_count()

    @property
    def clean(self) -> bool | None:
        if self.before is None or self.after is None:
            return None
        return self.after == self.before

    def report(self) -> None:
        t = smi("temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active")
        if t:
            print(f"  gpu      {t[0]}C, {float(t[1]):.0f} W, {t[2]} MHz at the end")
        if self.clean is None:
            print("  driver   not checked (no event log on this platform)")
        elif self.clean:
            print("  driver   no display-driver reset during this run")
        else:
            n = self.after - self.before
            print(
                f"  driver   *** {n} DISPLAY-DRIVER RESET(S) DURING THIS RUN ***\n"
                "           Every number above spans a driver reset and should not be quoted.\n"
                "           See ADR-0023: TdrDelay defaults to 2 seconds and this GPU also\n"
                "           drives the display, so a long-held device trips the watchdog."
            )


def main() -> int:
    n = tdr_count()
    print(f"  TDR events in the log: {n if n is not None else 'unknown'}")
    t = smi("name,driver_version,temperature.gpu,power.draw,power.limit,"
            "pcie.link.gen.current,pcie.link.width.current")
    if t:
        print(f"  {t[0]}, driver {t[1]}")
        print(f"  {t[2]}C, {float(t[3]):.0f} W of {float(t[4]):.0f} W, "
              f"PCIe gen {t[5]} x{t[6]}")
    else:
        print("  nvidia-smi did not answer -- the GPU may be lost")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
