from typing import List, Tuple

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import DensityMatrix, Statevector

# Operations whose name fully identifies them; anything else is compared by its definition.
_STANDARD_OPS = set(get_standard_gate_name_mapping())


class result_getter():
    '''
    Base class for result getters. A result getter turns a list of pubs,
    ``[(quantum_circuit, parameter_values), ...]``, into a DataFrame with one
    row per pub, in the same order.

    Primitive-based getters should not submit one job entry per pub. Pubs that
    share a circuit structure are grouped with :meth:`group_pubs`: each distinct
    circuit is transpiled once and submitted as a single primitive pub whose
    parameter values stack every pub that uses it.

    Result getters read qubits ``0, ..., num_qubits - 1`` of each circuit. A
    circuit may be wider (e.g. a reservoir whose encoding qubits are the last
    ones); its other qubits are not measured.
    '''

    def get_results(self, pubs: List[Tuple[QuantumCircuit, List[float]]]) -> pd.DataFrame:
        '''Get results for the given pubs, one row per pub.'''
        pass

    @staticmethod
    def group_pubs(pubs) -> List[Tuple[QuantumCircuit, List[int]]]:
        '''
        Group pubs by circuit structure.

        Two pubs share a group when their circuits are the same object or have
        identical instructions, qubits and parameter names, so they differ only
        in parameter values. Reservoirs rebuild an identical circuit for every
        pub, so grouping on structure rather than identity is what lets them
        share one transpilation and one primitive pub.

        Returns
        -------
        list of (circuit, indices)
            One entry per distinct circuit, in order of first appearance, with
            the positions in `pubs` that use it.
        '''
        groups = {}
        key_by_id = {}
        for i, (circ, _) in enumerate(pubs):
            if id(circ) not in key_by_id:
                key_by_id[id(circ)] = _circuit_key(circ)
            key = key_by_id[id(circ)]
            if key not in groups:
                groups[key] = (circ, [])
            groups[key][1].append(i)
        return list(groups.values())

    @staticmethod
    def parameter_matrix(circuit, pubs, indices, target=None) -> np.ndarray:
        '''
        Stack the parameter values of ``pubs[i] for i in indices`` into an array
        of shape ``(len(indices), num_parameters)``. Parameter values may be
        sequences ordered like ``circuit.parameters`` or dicts keyed by
        Parameter or parameter name.

        `target` is the transpiled circuit the values are bound to. Its
        parameters are matched by name, because transpiling can drop some
        (e.g. a diagonal gate right before a measurement is removed together
        with its parameter).
        '''
        names = [p.name for p in circuit.parameters]
        rows = []
        for i in indices:
            params = pubs[i][1]
            if isinstance(params, dict):
                by_name = {getattr(p, "name", p): v for p, v in params.items()}
                params = [by_name[name] for name in names]
            rows.append(np.asarray(params, dtype=float).reshape(-1))
        matrix = np.asarray(rows, dtype=float).reshape(len(indices), len(names))
        if target is None:
            return matrix
        column = {name: j for j, name in enumerate(names)}
        return matrix[:, [column[p.name] for p in target.parameters]]

    @staticmethod
    def exact_state(circ, params):
        '''Bind parameters and return the exact final state of the circuit.'''
        bound = circ.assign_parameters(params, inplace=False)
        # Statevector applies reset as a random projective measurement, so
        # circuits with mid-circuit resets (reservoirs) need a density matrix
        # to give exact, deterministic results.
        if "reset" in bound.count_ops():
            return DensityMatrix.from_instruction(bound)
        return Statevector.from_instruction(bound)

    @staticmethod
    def measured_circuit(circ, num_qubits) -> QuantumCircuit:
        '''
        Copy of `circ` with final measurements removed and qubits
        ``0, ..., num_qubits - 1`` measured into a new register ``meas``
        (classical bit i holds qubit i).
        '''
        measured = circ.remove_final_measurements(inplace=False)
        meas = ClassicalRegister(num_qubits, "meas")
        measured.add_register(meas)
        measured.measure(range(num_qubits), meas)
        return measured


def _circuit_key(circ):
    '''Hashable description of a circuit's structure, ignoring its name and metadata.'''
    return (
        circ.num_qubits,
        circ.num_clbits,
        tuple(p.name for p in circ.parameters),
        _instructions_key(circ),
    )


def _instructions_key(circ):
    items = []
    for inst in circ.data:
        op = inst.operation
        name, definition = op.name, None
        if op.name not in _STANDARD_OPS and getattr(op, "definition", None) is not None:
            # Composite gates (e.g. an appended ansatz) are compared by their
            # contents: equal names can hide different contents, and copies of
            # one circuit get distinct auto-generated names ("EfficientSU2-50").
            name, definition = None, _instructions_key(op.definition)
        items.append((
            name,
            tuple(circ.find_bit(q).index for q in inst.qubits),
            tuple(circ.find_bit(c).index for c in inst.clbits),
            tuple(str(p) for p in op.params),
            definition,
        ))
    return tuple(items)
