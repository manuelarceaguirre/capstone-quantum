from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator

from .pce_result_getter import pce_result_getter


class single_qubit_pauli_result_getter(pce_result_getter):
    """
    Result extraction utility for single-qubit Pauli expectation values.

    For every qubit i this computes <X_i>, <Y_i> and <Z_i>, i.e. the Bloch
    vector of each qubit's one-particle reduced density matrix. These are
    the features used by projected quantum kernels:
    https://quantum.cloud.ibm.com/docs/en/tutorials/projected-quantum-kernels

    The results are returned as a pandas DataFrame, where:

    - Rows correspond to circuit evaluations.
    - Columns are ``X_q0, ..., X_q{n-1}, Y_q0, ..., Z_q{n-1}``.

    Execution follows `pce_result_getter`: an Estimator primitive when one
    is provided, exact statevector-style evaluation when `estimator=None`.
    """

    def __init__(
        self,
        num_qubits,
        estimator: Estimator = None
    ):
        """
        Initialize the single-qubit Pauli result getter.

        Parameters
        ----------
        num_qubits : int
            Number of qubits used in the quantum circuit.

        estimator : Estimator
            Qiskit Estimator primitive used to evaluate expectation values.

            If set to None, the class computes exact expectation values
            directly from the quantum state.

            If the estimator is provided, the class will transpile the first circuit
            and assume that the same parameterized circuit structure is used for all entries in `pubs`.
        """

        self.estimator = estimator
        self.num_qubits = num_qubits

        # Qiskit Pauli labels are little-endian: qubit i is character n-1-i.
        self.pce_strings = [
            "I" * (num_qubits - 1 - i) + pauli + "I" * i
            for pauli in "XYZ"
            for i in range(num_qubits)
        ]
        self.ops = SparsePauliOp(self.pce_strings)

        # Readable column names aligned with self.pce_strings.
        self.columns = [
            f"{pauli}_q{i}"
            for pauli in "XYZ"
            for i in range(num_qubits)
        ]

    def get_results(self, pubs):
        """
        Evaluate <X_i>, <Y_i> and <Z_i> for every qubit of each circuit.

        Parameters
        ----------
        pubs : list
            List of tuples in the form:

            [
                (quantum_circuit, parameter_values),
                ...
            ]

        Returns
        -------
        pandas.DataFrame
            DataFrame of single-qubit expectation values.

            - Rows correspond to circuit evaluations.
            - Columns correspond to (Pauli, qubit) pairs.
        """

        results = super().get_results(pubs)
        results.columns = self.columns
        return results
