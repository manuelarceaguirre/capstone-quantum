from qiskit import transpile
from qiskit_ibm_runtime import Sampler

import numpy as np
import pandas as pd

from .result_getter import result_getter


class bitstring_probability_result_getter(result_getter):
    """
    Result extraction utility based on full computational basis sampling.

    Every qubit is measured in the computational basis and the probability
    of each of the 2^n bitstrings is returned as a feature.

    The results are returned as a pandas DataFrame, where:

    - Rows correspond to circuit evaluations.
    - Columns correspond to bitstrings in Qiskit order (qubit 0 is the
      rightmost character), from ``'0...0'`` to ``'1...1'``.

    The number of features grows as 2^num_qubits, so this is only practical
    for small circuits.

    The class supports two execution modes:

    1. Sampler-based execution:
       Uses a Qiskit Runtime Sampler primitive for hardware or simulator
       execution. Probabilities are shot-count frequencies.

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
        Initialize the bitstring probability result getter.

        Parameters
        ----------
        num_qubits : int
            Number of qubits used in the quantum circuit.

        sampler : Sampler
            Qiskit Sampler primitive used to sample bitstrings.

            If set to None, the class computes exact bitstring
            probabilities directly from the quantum state.

            If the sampler is provided, the class will transpile the first circuit
            and assume that the same parameterized circuit structure is used for all entries in `pubs`.

        shots : int
            Number of shots per circuit. If None, the sampler default is used.
        """

        self.sampler = sampler
        self.num_qubits = num_qubits
        self.shots = shots

        # Column j is the bitstring whose integer value is j.
        self.columns = [
            format(j, f"0{num_qubits}b")
            for j in range(2 ** num_qubits)
        ]

    def get_results(self, pubs):
        """
        Evaluate bitstring probabilities for a collection of circuits.

        Parameters
        ----------
        pubs : list
            List of tuples in the form:

            [
                (quantum_circuit, parameter_values),
                ...
            ]

            Qubits 0 to num_qubits - 1 are measured; the circuit may have more
            qubits, which are traced out. With a sampler, pubs that share a
            circuit structure are grouped: each distinct circuit is transpiled
            once and submitted as one sampler pub carrying the parameter values
            of every pub that uses it.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the probability of each bitstring.

            - Rows correspond to circuit evaluations, in the order of `pubs`.
            - Columns correspond to bitstrings.
        """

        values = np.zeros((len(pubs), 2 ** self.num_qubits))

        # Exact evaluation mode (no sampler provided).
        if self.sampler is None:

            qargs = list(range(self.num_qubits))

            for row, (circ, params) in enumerate(pubs):
                psi = self.exact_state(circ, params)

                # Indexed by the little-endian integer value of each bitstring.
                values[row] = psi.probabilities(qargs)

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

                bits = pub_result.data.meas

                # One set of counts per pub in the group; unobserved bitstrings stay zero.
                for k, row in enumerate(indices):
                    for value, count in bits.get_int_counts(loc=k).items():
                        values[row, value] = count / bits.num_shots

        return pd.DataFrame(values, columns=self.columns)
