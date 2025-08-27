# cpu_circuit_runner.py
# macOS / CPU-only Qiskit workflow:
# - circuit factory (modular, parameterized)
# - printing: diagrams, gate set, counts, ordered trace
# - optional transpile preview
# - Aer CPU simulation (no GPU options)

from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

try:
    # Newer API, present on recent qiskit-aer
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2  # type: ignore
    _HAS_AER_SAMPLER_V2 = True
except Exception:
    _HAS_AER_SAMPLER_V2 = False


# ---------- CPU-only simulator helpers ----------

def make_cpu_simulator(
    prefer: str = "matrix_product_state",
    shots: Optional[int] = None,
    max_parallel_threads: Optional[int] = None,
    precision: str = "double",
) -> AerSimulator:
    """
    Return a CPU-only AerSimulator for macOS.
    prefer: one of {'automatic','statevector','density_matrix','stabilizer',
                    'extended_stabilizer','matrix_product_state','unitary','superop'}
    """
    sim = AerSimulator(method=prefer, precision=precision)
    if max_parallel_threads is not None:
        sim.set_options(max_parallel_threads=max_parallel_threads)
    if shots is not None:
        sim._default_shots = int(shots)  # type: ignore[attr-defined]
    return sim


def run_on_cpu_sim(
    qc: QuantumCircuit,
    simulator: Optional[AerSimulator] = None,
    shots: Optional[int] = None,
    seed_simulator: Optional[int] = None,
):
    sim = simulator or make_cpu_simulator()
    run_kwargs = {}
    if shots is not None:
        run_kwargs["shots"] = int(shots)
    elif hasattr(sim, "_default_shots"):
        run_kwargs["shots"] = sim._default_shots  # type: ignore[attr-defined]
    if seed_simulator is not None:
        run_kwargs["seed_simulator"] = int(seed_simulator)
    job = sim.run(qc, **run_kwargs)
    return job.result()


# ---------- Circuit factory (modular, parameterized) ----------

@dataclass
class AnsatzConfig:
    n_qubits: int
    depth: int
    entanglement: str = "linear"   # 'linear' or 'ring'
    use_barriers: bool = True
    measure: bool = True


def build_layered_ry_cx_ansatz(cfg: AnsatzConfig) -> QuantumCircuit:
    """
    A transparent, easy-to-read layered ansatz:
    - Per-layer: RY(theta[q, layer]) on each qubit
    - Entangle with CX in a chosen pattern (linear or ring)
    - Optional barriers for legibility
    """
    qr = QuantumRegister(cfg.n_qubits, "q")
    cr = ClassicalRegister(cfg.n_qubits, "c") if cfg.measure else None
    qc = QuantumCircuit(qr, cr) if cr else QuantumCircuit(qr)

    thetas = [[Parameter(f"θ_{q}_{l}") for l in range(cfg.depth)] for q in range(cfg.n_qubits)]

    for l in range(cfg.depth):
        # 1) Single-qubit layer
        for q in range(cfg.n_qubits):
            qc.ry(thetas[q][l], q)
        if cfg.use_barriers:
            qc.barrier()

        # 2) Entanglement layer
        for q in range(cfg.n_qubits - 1):
            qc.cx(q, q + 1)
        if cfg.entanglement == "ring" and cfg.n_qubits > 2:
            qc.cx(cfg.n_qubits - 1, 0)

        if cfg.use_barriers:
            qc.barrier()

    if cfg.measure:
        qc.measure_all()

    return qc


# ---------- Introspection & printing ----------

def summarize_ops(qc: QuantumCircuit):
    """
    Qiskit Terra >= 0.25: Qubit/Clbit no longer expose `.index`.
    We derive stable indices from the circuit order:
      - q index = position in qc.qubits
      - c index = position in qc.clbits
    Falls back to qc.find_bit(...) if needed.
    """
    from collections import Counter

    # Build fast lookup maps once
    q_index = {bit: i for i, bit in enumerate(qc.qubits)}
    c_index = {bit: i for i, bit in enumerate(qc.clbits)}

    def qids_from(qargs):
        ids = []
        for q in qargs:
            if q in q_index:
                ids.append(q_index[q])
            else:
                # Rare fallback (e.g., if q not in qc.qubits for any reason)
                ids.append(qc.find_bit(q).index)
        return ids

    def cids_from(cargs):
        ids = []
        for c in cargs:
            if c in c_index:
                ids.append(c_index[c])
            else:
                ids.append(qc.find_bit(c).index)
        return ids

    op_counts = Counter()
    ordered_trace = []
    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        name = inst.name
        op_counts[name] += 1
        qids = qids_from(qargs)
        cids = cids_from(cargs)
        params = list(inst.params) if getattr(inst, "params", None) else []
        ordered_trace.append((idx, name, qids, cids, params))

    # Partition unitary vs non-unitary (common names)
    nonunitary = {"measure", "barrier", "reset", "delay", "snapshot"}
    ops_nonunitary = {k: v for k, v in op_counts.items() if k in nonunitary}
    gate_set_unitary = sorted(k for k in op_counts.keys() if k not in nonunitary)

    return dict(
        op_counts=dict(sorted(op_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        gate_set_unitary=gate_set_unitary,
        ops_nonunitary=dict(sorted(ops_nonunitary.items(), key=lambda kv: (-kv[1], kv[0]))),
        ordered_trace=ordered_trace,
    )

def print_circuit_report(
    title: str,
    qc: QuantumCircuit,
    show_qasm: bool = False,
    max_trace: int = 80,
):
    """
    Print a human-friendly report of the circuit: diagram, summary, gate set,
    counts, and ordered trace (first N ops).
    """
    print("=" * 80)
    print(f"[{title}]  qubits={qc.num_qubits}  clbits={qc.num_clbits}  depth={qc.depth()}  size={qc.size()}")
    print("-" * 80)
    # ASCII diagram
    print(qc.draw(output="text"))
    print("-" * 80)

    info = summarize_ops(qc)

    # Gate set & counts
    print("Gate set (unitary only):", ", ".join(info["gate_set_unitary"]) or "—")
    print("Operation counts (all):")
    for name, cnt in info["op_counts"].items():
        print(f"  {name:>16s} : {cnt}")

    if info["ops_nonunitary"]:
        print("Non-unitary ops:")
        for name, cnt in info["ops_nonunitary"].items():
            print(f"  {name:>16s} : {cnt}")

    # Ordered trace (first N ops)
    print("-" * 80)
    print(f"Ordered op trace (first {max_trace}):")
    for (idx, name, qids, cids, params) in info["ordered_trace"][:max_trace]:
        if params:
            pstr = ", ".join(str(p) for p in params)
            print(f"  [{idx:04d}] {name:<18s} q={qids} c={cids} params=[{pstr}]")
        else:
            print(f"  [{idx:04d}] {name:<18s} q={qids} c={cids}")

    # (Optional) Raw OpenQASM (can be verbose)
    if show_qasm:
        try:
            qasm_str = qc.qasm()  # may be deprecated on some versions; guarded by try/except
            print("-" * 80)
            print("OpenQASM 2.0:")
            print(qasm_str)
        except Exception as e:
            print("(qasm() not available on this Qiskit version)")
            print(f"Reason: {e}")


def transpile_for_aer(qc: QuantumCircuit, backend: AerSimulator, optimization_level: int = 2) -> QuantumCircuit:
    """
    Transpile the circuit for Aer (so you can inspect the actual primitive gates used).
    """
    from qiskit import transpile
    tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
    return tqc


# ---------- Optional: Aer Sampler V2 on CPU ----------

def make_cpu_sampler_v2(
    prefer: str = "matrix_product_state",
    shots: Optional[int] = 4096,
    max_parallel_threads: Optional[int] = None,
    precision: str = "double",
):
    if not _HAS_AER_SAMPLER_V2:
        return make_cpu_simulator(prefer=prefer, shots=shots, max_parallel_threads=max_parallel_threads, precision=precision)
    sim = make_cpu_simulator(prefer=prefer, shots=shots, max_parallel_threads=max_parallel_threads, precision=precision)
    return AerSamplerV2(backend=sim, default_shots=shots)  # type: ignore

def bind_params_compat(circ: QuantumCircuit, param_map: dict) -> QuantumCircuit:
    """
    Works across Qiskit versions:
      - If .bind_parameters exists, use it (returns a new circuit).
      - Otherwise, fall back to .assign_parameters(..., inplace=False).
    """
    if not param_map:
        return circ
    if hasattr(circ, "bind_parameters"):
        return circ.bind_parameters(param_map)  # newer/alternate API
    # Older/newer Terra variants
    return circ.assign_parameters(param_map, inplace=False)


# ---------- Example usage ----------

if __name__ == "__main__":
    # 1) Build a transparent, layered RY+CX circuit
    cfg = AnsatzConfig(n_qubits=6, depth=3, entanglement="linear", use_barriers=True, measure=True)
    qc = build_layered_ry_cx_ansatz(cfg)

    # 2) Print the ORIGINAL circuit (diagram + gates used)
    print_circuit_report("ORIGINAL CIRCUIT", qc, show_qasm=False, max_trace=120)

    # 3) Make a CPU-only simulator and transpile to see the actual basis gates Aer will use
    sim = make_cpu_simulator(prefer="matrix_product_state", shots=4096, max_parallel_threads=0)
    tqc = transpile_for_aer(qc, sim, optimization_level=2)

    # 4) Print the TRANSPILED circuit (often reveals decompositions/basis gates)
    print_circuit_report("TRANSPILED FOR AER", tqc, show_qasm=False, max_trace=120)

    # 5) (Optional) Bind parameters for deterministic sampling (here: set all θ = 0.3)
    #    You can customize this as needed; leaving params unbound is also fine for state simulation.
    params = {p: 0.3 for p in tqc.parameters}
    tqc = bind_params_compat(tqc, params)  # no-op if params is empty

    # 6) Run on CPU and show a sample of counts
    result = run_on_cpu_sim(tqc, simulator=sim)  # uses default shots from sim (4096)
    counts = result.get_counts(tqc)
    # Print top-k outcomes
    print("-" * 80)
    print("Result (sample of measurement outcomes):")
    top_items = list(sorted(counts.items(), key=lambda kv: -kv[1]))[:10]
    for bitstr, n in top_items:
        print(f"  {bitstr} : {n}")