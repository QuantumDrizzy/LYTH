#!/usr/bin/env python3
"""ADR-0029 P5: the outside oracle for circuits LYTH runs on the GPU.

    python tools/qgpu_oracle.py CIRCUIT.json OUT.json

Reads a circuit as `crates/lyth/tests/circuits.rs` writes it -- the number of qubits and a list of
gates, each with its 2x2 matrix in f64 -- builds the same circuit in Qiskit from |0...0>, and
writes Qiskit's `Statevector` (f64) as `{"re": [...], "im": [...]}`.

Nothing here shares code with LYTH: the gates are applied by Qiskit's own `UnitaryGate`, its
`.control(1)` and its `swap`. Qubit order is little-endian in both (bit q of the index is qubit q),
so no reordering sits between them.
"""

import json
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def matrix(m):
    return np.array([[complex(*m[0]), complex(*m[1])], [complex(*m[2]), complex(*m[3])]])


def main(src, dst):
    spec = json.load(open(src))
    qc = QuantumCircuit(spec["qubits"])
    for g in spec["gates"]:
        if g["kind"] == "u":
            qc.append(UnitaryGate(matrix(g["m"])), [g["q"]])
        elif g["kind"] == "cu":
            qc.append(UnitaryGate(matrix(g["m"])).control(1), [g["c"], g["t"]])
        elif g["kind"] == "swap":
            qc.swap(g["a"], g["b"])
        else:
            sys.exit(f"unknown gate kind {g['kind']!r}")
    psi = Statevector.from_instruction(qc).data
    json.dump({"re": psi.real.tolist(), "im": psi.imag.tolist()}, open(dst, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
