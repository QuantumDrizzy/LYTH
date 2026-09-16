#!/usr/bin/env python3
"""Compile the hand-written CUDA C++ to PTX, without a host compiler.

    python bench/nvrtc.py                    # writes bench/cuda/handwritten.ptx

`nvcc -ptx` needs a host C++ compiler even when it emits no host code, and on this machine the
installed MSVC (14.51) refuses CUDA 13.0 from inside its own STL headers:

    error STL1002: Unexpected compiler version, expected CUDA 13.2 or newer.

`-allow-unsupported-compiler` does not help, because the failure is a `static_assert` in the
STL rather than nvcc's own version gate. NVRTC compiles CUDA C++ to PTX with no host compiler
in the path at all, which removes a dependency the comparison never needed: the `.cu` here
includes nothing.

Driven through `ctypes`, the same way the generated LYTH bindings talk to the driver. That is
not a coincidence worth hiding -- it means the hand-written kernel and the generated one reach
the GPU by exactly the same route, so a difference between them is a difference between the
kernels.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
CUDA = pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")


def load_nvrtc() -> ctypes.CDLL:
    for cand in sorted((CUDA / "bin" / "x64").glob("nvrtc64_*.dll"), reverse=True):
        if ".alt." in cand.name:
            continue
        return ctypes.WinDLL(str(cand))
    sys.exit(f"no nvrtc found under {CUDA / 'bin' / 'x64'}")


def compile_ptx(
    source: str, name: str, arch: str = "compute_120", extra: tuple[str, ...] = ()
) -> str:
    """CUDA C++ in, PTX out. Raises with the compiler log on failure.

    `extra` is for options a *comparison* needs rather than options that make the kernel
    fast. The one in use is `--fmad=false` (ADR-0021 step 5): LYTH emits a multiply and an
    add as two instructions on purpose, since one rounding is a different answer from two
    (ADR-0010), so an nvrtc left to contract them is computing something else and a timing
    against it would be a timing against a different function.
    """
    nvrtc = load_nvrtc()

    def check(code: int, what: str) -> None:
        if code != 0:
            nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"{what}: {nvrtc.nvrtcGetErrorString(code).decode()}")

    prog = ctypes.c_void_p()
    check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), source.encode(), name.encode(), 0, None, None
        ),
        "nvrtcCreateProgram",
    )

    opts = [f"--gpu-architecture={arch}".encode(), b"-default-device"]
    opts += [o.encode() for o in extra]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)

    # The log is fetched whether or not it compiled: warnings are worth seeing.
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    text = log.value.decode().strip()
    if rc != 0:
        raise RuntimeError(f"nvrtc failed:\n{text}")
    if text:
        print(f"  nvrtc log:\n{text}", file=sys.stderr)

    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetPTX(prog, buf)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.value.decode()


def main() -> int:
    src = HERE / "cuda" / "handwritten.cu"
    out = HERE / "cuda" / "handwritten.ptx"
    ptx = compile_ptx(src.read_text(encoding="utf-8"), src.name)
    out.write_text(ptx, encoding="utf-8", newline="\n")
    entries = [l.split("(")[0].split()[-1] for l in ptx.splitlines() if ".visible .entry" in l]
    print(f"  {out.relative_to(HERE.parent)}  {len(ptx)} bytes")
    for e in entries:
        print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
