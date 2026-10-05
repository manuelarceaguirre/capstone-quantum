from .encoding_strategy import encoding_strategy

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Estimator
from qiskit.circuit.library import efficient_su2
from sklearn.preprocessing import MinMaxScaler
from typing import List
import pandas as pd
import numpy as np


class su_encoding_strategy(encoding_strategy):
    """
    Quantum data encoding strategy based on parameterized unitary circuits.

    This encoding strategy transforms classical tabular data into parameter
    assignments for an EfficientSU2 quantum circuit ansatz.

    The workflow consists of:

    1. Scaling classical input data to the interval [0, 2π]
    2. Mapping scaled features to circuit parameters
    3. Generating pubs of the form:

       (QuantumCircuit, parameter_values)

    These pubs (Primitive Unitary Blocks) can later be submitted to qiskit primitives
    to estimate expectation values or bitstring probabilities.

    Notes
    -----
    - Uses the Qiskit `efficient_su2` ansatz.
    - Automatically pads feature vectors with zeros if the number of
      classical features is smaller than the number of circuit parameters.
    """

    def __init__(self, 
                 n_qubits: int = None,
                 reps: int = 1,
                 su2_gates: List[str] = ['ry', 'rx'],
                 entanglement: str = 'full'):
        """
        Initialize the unitary encoding strategy. The maximum number of features that can be
        encoded is determined by the number of parameters in the EfficientSU2 circuit, which
        is n_qubits * reps * len(su2_gates).

        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits used in the quantum circuit.

        reps : int, default=1
            Number of repeated layers in the EfficientSU2 circuit.

        su2_gates : List[str], default=['ry', 'rx']
            List of single-qubit rotation gates used in the ansatz.

            Options include:
            - 'rx'
            - 'ry'
            - 'rz'

        entanglement : str, default='full'
            Entanglement pattern used in the EfficientSU2 circuit.

            Options include:
            - 'full'
            - 'linear'
            - 'circular'

        """

        self.n_qubits = n_qubits
        self.reps = reps
        self.su_2_gates = su2_gates
        self.entanglement = entanglement

    def generate_pubs(
        self,
        data: pd.DataFrame
    ) -> List[tuple[QuantumCircuit, List[float]]]:
        """
        Generate publication tuples for quantum execution.

        This method converts classical input data into parameter assignments
        for a parameterized EfficientSU2 quantum circuit.

        The generated output is a list of tuples of the form:

            (quantum_circuit, parameter_values)

        where each parameter vector corresponds to one row of the input data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input feature matrix containing classical data samples.

            - Rows correspond to observations/samples.
            - Columns correspond to classical features.

        Returns
        -------
        List[tuple[QuantumCircuit, List[float]]]
            List of pubs ready for a qiskit primitive.

            Each tuple contains:
            - QuantumCircuit
                The parameterized EfficientSU2 circuit.
            - List[float]
                Parameter values assigned to the circuit.

        Notes
        -----
        Data preprocessing steps include:

        1. Min-max scaling to the interval [0, 1]
        2. Rescaling to angular values in [0, 2π]

        If the number of classical features is smaller than the number
        of circuit parameters, zero-padding is automatically applied.
        """

        # Construct the parameterized EfficientSU2 circuit ansatz.
        circuit = efficient_su2(
            self.n_qubits,
            reps=self.reps,
            su2_gates=self.su_2_gates,
            entanglement=self.entanglement,
        )

        # Scale input data into the interval [0, 1].
        scaled_data = MinMaxScaler().fit_transform(data.values)

        # Rescale normalized values into angular encoding range [0, 2π].
        scaled_data = 2 * np.pi * scaled_data

        # Store generated publication tuples.
        pubs = []

        # Total number of circuit parameters.
        n_params = circuit.num_parameters

        # Zero-pad feature vectors if there are fewer classical features
        # than circuit parameters.
        if scaled_data.shape[1] < n_params:
            scaled_data = np.hstack([
                scaled_data,
                np.zeros((
                    scaled_data.shape[0],
                    n_params - scaled_data.shape[1]
                ))
            ])

        # Create one publication tuple per observation.
        for row in scaled_data:
            pubs.append((circuit, row))

        return pubs