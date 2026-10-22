from .encoding_strategy import encoding_strategy

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from sklearn.preprocessing import MinMaxScaler
from typing import List, Sequence
import pandas as pd
import numpy as np


# Two-qubit Pauli rotations available as Hamiltonian couplings.
_PAULI_GATES = {
    "XX": QuantumCircuit.rxx,
    "YY": QuantumCircuit.ryy,
    "ZZ": QuantumCircuit.rzz,
}


class heisenberg_encoding_strategy(encoding_strategy):
    """
    Quantum data encoding strategy based on Trotterized Hamiltonian evolution.

    Each classical feature sets the coupling strength of a nearest-neighbour
    two-qubit Pauli interaction, and the resulting Hamiltonian

        H(x) = sum_k x_k * sum_{P in paulis} P_{q_k} P_{q_k + 1}

    is evolved for unit time with a first-order Trotter decomposition.

    With the default ``paulis=("ZZ", "XX", "YY")`` this is the Heisenberg
    ansatz (https://arxiv.org/pdf/2509.17715, following the Qiskit tutorial
    implementation). With ``paulis=("ZZ",)`` it reduces
    to a ZZ Pauli feature map style evolution on a linear chain.

    The workflow consists of:

    1. Preparing a fixed Haar random single-qubit state on every qubit
    2. Scaling classical input data to the interval [0, 2π]
    3. Mapping scaled features to bond couplings of the Trotter circuit
    4. Generating pubs of the form:

       (QuantumCircuit, parameter_values)

    Notes
    -----
    - Feature k acts on the bond (q, q + 1) with q = k mod (n_qubits - 1),
      so by default n_qubits = n_features + 1 and every feature gets its
      own bond. Fewer qubits make features wrap around the chain.
    - Within a Trotter step, even bonds are applied before odd bonds
      (brickwork), and every other step is reversed to cancel part of the
      first-order Trotter error.
    - Automatically pads feature vectors with zeros if the number of
      classical features is smaller than n_features.
    """

    def __init__(self,
                 n_features: int,
                 n_qubits: int = None,
                 num_trotter_steps: int = 6,
                 paulis: Sequence[str] = ("ZZ", "XX", "YY"),
                 seed: int = 42,
                 scale: bool = True):
        """
        Initialize the Heisenberg encoding strategy.

        Parameters
        ----------
        n_features : int
            Maximum number of classical features encoded by the circuit.

        n_qubits : int, optional
            Number of qubits in the chain. Defaults to n_features + 1.
            Must be at least 2.

        num_trotter_steps : int, default=6
            Number of first-order Trotter steps used to evolve H(x).

        paulis : Sequence[str], default=("ZZ", "XX", "YY")
            Two-qubit Pauli couplings applied on every bond, in order.

            Options include:
            - 'XX'
            - 'YY'
            - 'ZZ'

        seed : int, default=42
            Seed for the Haar random single-qubit initial unitaries.

        scale : bool, default=True
            If True, min-max scale each batch passed to generate_pubs (statistics
            come from that batch). Set False when the data is already scaled to
            [0, 1], e.g. with causal statistics to avoid look-ahead.
        """

        if n_qubits is None:
            n_qubits = n_features + 1
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")

        unknown = set(paulis) - set(_PAULI_GATES)
        if unknown:
            raise ValueError(
                f"Unsupported paulis {sorted(unknown)}; "
                f"choose from {sorted(_PAULI_GATES)}"
            )

        self.n_features = n_features
        self.n_qubits = n_qubits
        self.num_trotter_steps = num_trotter_steps
        self.paulis = tuple(paulis)
        self.seed = seed
        self.scale = scale

        # Fixed Haar random initial state, shared by every sample.
        rng = np.random.default_rng(seed=seed)
        self.unitaries = [
            UnitaryGate(random_unitary(2, seed=int(rng.integers(0, 2**32))))
            for _ in range(n_qubits)
        ]

        self.circuit = self._build_circuit()

    def _trotter_step(self, parameters: ParameterVector) -> QuantumCircuit:
        """Build one first-order Trotter step of exp(-i H(x))."""

        n_bonds = self.n_qubits - 1
        step = QuantumCircuit(self.n_qubits)

        # Group features by bond parity so even bonds run before odd bonds.
        even = [k for k in range(self.n_features) if (k % n_bonds) % 2 == 0]
        odd = [k for k in range(self.n_features) if (k % n_bonds) % 2 == 1]

        for group in (even, odd):
            for pauli in self.paulis:
                for k in group:
                    q = k % n_bonds
                    _PAULI_GATES[pauli](
                        step,
                        2 * parameters[k] / self.num_trotter_steps,
                        q,
                        q + 1,
                    )

        return step

    def _build_circuit(self) -> QuantumCircuit:
        """Build the full parameterized encoding circuit."""

        parameters = ParameterVector("x", self.n_features)
        circuit = QuantumCircuit(self.n_qubits)

        # Haar random single-qubit initial state.
        for i, unitary in enumerate(self.unitaries):
            circuit.append(unitary, [i])

        # Trotterized Hamiltonian evolution, reversing every other step.
        step = self._trotter_step(parameters)
        for s in range(self.num_trotter_steps):
            circuit.compose(
                step if s % 2 == 0 else step.reverse_ops(),
                inplace=True,
            )

        return circuit

    def generate_pubs(
        self,
        data: pd.DataFrame
    ) -> List[tuple[QuantumCircuit, List[float]]]:
        """
        Generate publication tuples for quantum execution.

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
                The parameterized Heisenberg evolution circuit.
            - List[float]
                Parameter values assigned to the circuit.
        """

        if data.shape[1] > self.n_features:
            raise ValueError(
                f"data has {data.shape[1]} features but the circuit "
                f"encodes at most n_features={self.n_features}"
            )

        # Scale input data into the interval [0, 1].
        if self.scale:
            scaled_data = MinMaxScaler().fit_transform(data.values)
        else:
            scaled_data = np.asarray(data.values, dtype=float)

        # Rescale normalized values into angular encoding range [0, 2π].
        scaled_data = 2 * np.pi * scaled_data

        # Zero-pad feature vectors if there are fewer classical features
        # than circuit parameters.
        if scaled_data.shape[1] < self.n_features:
            scaled_data = np.hstack([
                scaled_data,
                np.zeros((
                    scaled_data.shape[0],
                    self.n_features - scaled_data.shape[1]
                ))
            ])

        # Create one publication tuple per observation.
        return [(self.circuit, row) for row in scaled_data]
