from qiskit import transpile
from qiskit_ibm_runtime import Sampler

import numpy as np
import pandas as pd

from .result_getter import result_getter


class single_bit_sampler_result_getter(result_getter):
    """
    Result extraction utility based on single-bit computational basis sampling.

    Every qubit is measured in the computational basis and the expectation
    value of each measured bit is returned independently, i.e. the
    probability of observing ``1`` on that qubit:

        E[b_i] = P(b_i = 1)

    The results are returned as a pandas DataFrame, where:

    - Rows correspond to circuit evaluations.
    - Columns correspond to qubits (``q0``, ``q1``, ...).

    The class supports two execution modes:

    1. Sampler-based execution:
       Uses a Qiskit Runtime Sampler primitive for hardware or simulator
       execution.

    2. Exact statevector-style evaluation:
       Triggered when `sampler=None`.
    """

    def __init__(
        self,
        num_qubits,
        sampler: Sampler = None,
        shots: int = None
    ):
        """
        Initialize the single-bit sampler result getter.

        Parameters
        ----------
        num_qubits : int
            Number of qubits used in the quantum circuit.

        sampler : Sampler
            Qiskit Sampler primitive used to sample bitstrings.

            If set to None, the class computes exact single-qubit
            probabilities directly from the quantum state.

            If the sampler is provided, the class will transpile the first circuit
            and assume that the same parameterized circuit structure is used for all entries in `pubs`.

        shots : int
            Number of shots per circuit. If None, the sampler default is used.
        """

        self.sampler = sampler
        self.num_qubits = num_qubits
        self.shots = shots

        # One feature column per measured qubit.
        self.columns = [f"q{i}" for i in range(num_qubits)]

    def get_results(self, pubs):
        """
        Evaluate per-qubit computational basis expectation values.

        Parameters
        ----------
        pubs : list
            List of tuples in the form:

            [
                (quantum_circuit, parameter_values),
                ...
            ]

            Qubits 0 to num_qubits - 1 are measured; the circuit may have more
            qubits. With a sampler, pubs that share a circuit structure are
            grouped: each distinct circuit is transpiled once and submitted as
            one sampler pub carrying the parameter values of every pub that
            uses it.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing P(b_i = 1) for each qubit.

            - Rows correspond to circuit evaluations, in the order of `pubs`.
            - Columns correspond to qubits.
        """

        values = np.empty((len(pubs), self.num_qubits))

        # Exact evaluation mode (no sampler provided).
        if self.sampler is None:

            for row, (circ, params) in enumerate(pubs):
                psi = self.exact_state(circ, params)

                # Marginal probability of measuring 1 on each qubit.
                values[row] = [psi.probabilities([i])[1] for i in range(self.num_qubits)]

        else:
            # Sampler-based execution mode: one sampler pub per distinct circuit.

            groups = self.group_pubs(pubs)
            sampler_pubs = []

            for circ, indices in groups:
                # Measure qubits 0..num_qubits-1 into "meas"; classical bit i holds qubit i
                # through any transpiled layout.
                transpiled = transpile(
                    self.measured_circuit(circ, self.num_qubits), self.sampler.backend()
                )
                sampler_pubs.append((transpiled, self.parameter_matrix(circ, pubs, indices, transpiled)))

            circuit_results = (
                self.sampler
                .run(pubs=sampler_pubs, shots=self.shots)
                .result()
            )

            for (_, indices), pub_result in zip(groups, circuit_results):

                # Shape (num_pubs_in_group, shots, num_qubits); last axis index i is qubit i.
                bits = pub_result.data.meas.to_bool_array(order="little")
                values[indices] = bits.mean(axis=1)

        return pd.DataFrame(values, columns=self.columns)
