#!/usr/bin/env python3
"""Measure achieved peak TFLOPS (SGEMM/HGEMM) and DRAM read BW for lyth machine files.

    python tools/peak_probe.py
    python tools/peak_probe.py --out fixtures/machine/meas-peak-sm_120.json

Writes a lyth-machine-measurement-compatible JSON plus peak_tflops for machine update.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def timed_cuda(fn, reps: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def dram_read_gbs(n_bytes: int = 1 << 30) -> float:
    n = n_bytes // 4
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    x.fill_(1.0)
    torch.cuda.synchronize()

    def run():
        # force a full read; sum is a stand-in for streaming load
        return float(x.sum())

    secs = timed_cuda(run, reps=20, warmup=5)
    return (n_bytes / secs) / 1e9


def gemm_tflops(dtype: torch.dtype, n: int = 8192, reps: int = 30) -> tuple[float, float]:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    # warm cublas
    c = a @ b
    torch.cuda.synchronize()
    del c

    def run():
        return a @ b

    secs = timed_cuda(run, reps=reps, warmup=10)
    # 2*n^3 FLOPs for GEMM
    flops = 2.0 * (n**3)
    tflops = (flops / secs) / 1e12
    return tflops, secs * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--machine-id", default="sm_120")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{props.major}{props.minor}"
    print(f"device: {props.name} {arch}  SMs={props.multi_processor_count}")

    bw = dram_read_gbs()
    print(f"dram read (torch sum): {bw:.2f} GB/s")

    fp32_tflops, fp32_ms = gemm_tflops(torch.float32, n=8192)
    print(f"SGEMM 8192: {fp32_tflops:.2f} TFLOP/s  ({fp32_ms:.2f} ms/iter)")

    fp16_tflops, fp16_ms = gemm_tflops(torch.float16, n=8192)
    print(f"HGEMM 8192: {fp16_tflops:.2f} TFLOP/s  ({fp16_ms:.2f} ms/iter)")

    # Ridge for lyth uses FP32-class peak unless noted; keep both.
    peak_for_ridge = fp32_tflops
    ridge = peak_for_ridge * 1e3 / bw if bw > 0 else 0.0
    print(f"ridge (fp32_peak*1e3/dram): {ridge:.1f} flop/byte")

    payload = {
        "schema": "lyth-machine-measurement/0.1",
        "machine_id": args.machine_id,
        "source": f"lyth/tools/peak_probe.py torch {torch.__version__}",
        "levels": [{"name": "dram", "bandwidth_gbs": bw}],
        "peak_tflops_fp32": fp32_tflops,
        "peak_tflops_fp16": fp16_tflops,
        "peak_tflops": peak_for_ridge,
        "arch": arch,
        "device": props.name,
        "notes": [
            "peak_tflops is achieved cuBLAS GEMM, not marketing datasheet.",
            "Use peak_tflops_fp32 for ridge unless the kernel is tensor-core dominant.",
        ],
    }

    text = json.dumps(payload, indent=2) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
