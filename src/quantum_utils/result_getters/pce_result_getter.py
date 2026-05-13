from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

import numpy as np
from math import comb, exp
from typing import List
import networkx as nx

from itertools import combinations

import pandas as pd

from .result_getter import result_getter


def generate_pauli_strings(
    num_qubits: int,
    num_features: int,
    num_operators: int
) -> List[str]:
    """
    Generate a collection of Pauli strings for Pauli Correlation Encoding (PCE).

    This function creates k-body Pauli operators across an n-qubit system
    using combinations of the Pauli matrices X, Y, and Z. Each generated
    string contains exactly `num_operators` non-identity Pauli operators,
    with all remaining qubits assigned the identity operator 'I'.

    The generated strings are commonly used as observables for extracting
    quantum feature representations from parameterized quantum circuits.

    Notes
    -----
    This implementation is adapted from the
    quantum-optimization-algorithms repository:
    https://github.com/SMU-Quantum/quantum-optimization-algorithms

    Copyright (c) 2025 Monit Sharma
    Licensed under the MIT License.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the quantum system.

    num_features : int
        Number of Pauli strings (features) to generate.

    num_operators : int
        Number of non-identity Pauli operators per string
        (i.e., the correlation order or k-body interaction size).

    Returns
    -------
    List[str]
        A list of Pauli strings such as:
        ['XXII', 'YYII', 'ZZII', ...]

    Raises
    ------
    ValueError
        If the requested number of operators exceeds the number of qubits.

    ValueError
        If the requested number of features exceeds the maximum number
        of unique Pauli strings that can be generated.
    """

    # This file is copied from quantum-optimization-algorithms
    # (https://github.com/SMU-Quantum/quantum-optimization-algorithms)
    # Copyright (c) 2025 Monit Sharma
    # Licensed under the MIT License

    # Rename inputs locally for readability and alignment with the
    # original implementation notation.
    n = num_qubits
    m = num_features
    k = num_operators

    # Ensure the correlation order does not exceed the number of qubits.
    if k > n:
        raise ValueError("k cannot be greater than n")

    # Maximum number of possible Pauli strings:
    # 3 Pauli operator choices (X, Y, Z) multiplied by
    # all k-qubit combinations.
    max_pauli_strings = 3 * comb(n, k)

    # Validate that the requested number of features is feasible.
    if m > max_pauli_strings:
        raise ValueError(
            f"The maximum number of Pauli strings that can be "
            f"generated for n={n} and k={k} is "
            f"{max_pauli_strings}. The requested m={m} exceeds "
            f"this limit."
        )

    # Define the Pauli operators used in feature construction.
    pauli_ops = ['X', 'Y', 'Z']

    # Generate all unique qubit position combinations for the
    # specified correlation order k.
    positions = list(combinations(range(n), k))

    # Store the generated Pauli strings.
    pauli_strings = []

    # Construct Pauli strings for every operator type and
    # qubit-position combination.
    for op in pauli_ops:
        for pos in positions:

            # Initialize with identity operators on all qubits.
            pauli_string = ['I'] * n

            # Insert the selected Pauli operator at the chosen positions.
            for i in pos:
                pauli_string[i] = op

            # Convert list representation to string form.
            pauli_strings.append(''.join(pauli_string))

    # Return only the requested number of features.
    return pauli_strings[:m]


class pce_result_getter(result_getter):
    """
    Result extraction utility for Pauli Correlation Encoding (PCE).

    This class computes expectation values of predefined Pauli observables
    for a collection of parameterized quantum circuits.

    The expectation values serve as quantum feature representations and are
    returned as a pandas DataFrame, where:

    - Rows correspond to circuit evaluations.
    - Columns correspond to Pauli feature strings.

    The class supports two execution modes:

    1. Estimator-based execution:
       Uses a Qiskit Runtime Estimator primitive for hardware or simulator
       execution.

    2. Exact statevector-style evaluation:
       Triggered when `estimator=None`.
    """

    def __init__(
        self,
        num_qubits,
        num_features,
        num_operators,
        estimator: Estimator = None
    ):
        """
        Initialize the Pauli Correlation Encoding result getter.

        Parameters
        ----------
        num_qubits : int
            Number of qubits used in the quantum circuit.

        num_features : int
            Number of Pauli feature observables to generate.

        num_operators : int
            Number of non-identity Pauli operators per feature string.

        estimator : Estimator
            Qiskit Estimator primitive used to evaluate expectation values.

            If set to None, the class attempts to compute exact
            expectation values directly from the quantum state.

            If the estimator is provided, the class will transpile the first circuit
            and assume that the same parameterized circuit structure is used for all entries in `pubs`.
        """

        # Store the estimator backend or primitive.
        self.estimator = estimator
        self.num_qubits = num_qubits

        # Generate Pauli feature strings used as observables.
        self.pce_strings = generate_pauli_strings(
            num_qubits,
            num_features,
            num_operators
        )

        # Convert the Pauli strings into a SparsePauliOp object
        # for efficient expectation value computation.
        self.ops = SparsePauliOp(self.pce_strings)

    def get_results(self, pubs):
        """
        Evaluate Pauli expectation values for a collection of circuits.

        Parameters
        ----------
        pubs : list
            List of tuples in the form:

            [
                (quantum_circuit, parameter_values),
                ...
            ]

            where:

            - quantum_circuit : QuantumCircuit
                Parameterized quantum circuit.

            - parameter_values : dict or sequence
                Values assigned to the circuit parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing expectation values for each Pauli feature.

            - Rows correspond to circuit evaluations.
            - Columns correspond to generated Pauli strings.
        """

        # Exact evaluation mode (no estimator provided).
        if self.estimator is None:

            exact_results = []

            # Evaluate expectation values directly from the quantum state.
            for circ, params in pubs:

                # Bind parameters to the circuit.
                bound = circ.assign_parameters(params, inplace=False)
                psi = Statevector.from_instruction(bound)

                # Compute expectation value for every Pauli observable.
                for op in self.ops:
                    exact_results.append(
                        psi.expectation_value(op).real
                    )

        else:
            # Estimator-based execution mode.

            estimator_pubs = []

            # Create one estimator job entry per observable.
            for circ, params in pubs:
                transpiled = transpile(pubs[0][0], self.estimator.backend())
                for op in self.ops:
                    estimator_pubs.append((transpiled, op.apply_layout(transpiled.layout), params))

            # Execute all observables through the estimator.
            circuit_results = (
                self.estimator
                .run(pubs=estimator_pubs)
                .result()
            )

        # Initialize results DataFrame with Pauli string columns.
        results = pd.DataFrame(columns=self.pce_strings, dtype=float)

        index = 0

        # Populate DataFrame row-by-row.
        for row in range(len(pubs)):

            for column in self.pce_strings:

                if self.estimator is None:
                    results.loc[row, column] = exact_results[index]

                else:
                    results.loc[row, column] = circuit_results[index].data.evs

                index += 1

        return results