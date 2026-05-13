from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

import pandas as pd
from typing import List

from src.quantum_utils.encoding_strategy.encoding_strategy import encoding_strategy
from src.quantum_utils.result_getters.result_getter import result_getter


def quantum_reservoir(data : pd.DataFrame,
                      series_indicies : List[List[int]],
                      encoding_strategy : encoding_strategy,
                      ansatz : QuantumCircuit,
                      result_getter : result_getter,
                      return_circuit : bool = False
                      ):
    """
    Constructs and evaluates a quantum reservoir circuit for sequential time series data.

    This function builds a reservoir-style quantum circuit by repeatedly:
        1. Encoding a feature vector into a quanutm state,
        2. Applying a shared, untrained ansatz circuit to the encoding and hidden qubits,
        3. Resetting the encoding qubits,
    across all indices in each input sequence.

    The workflow is designed for sequential or temporal modeling tasks where
    each sequence in `series_indicies` represents an ordered collection of
    observations from `data`.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing the classical feature vectors to encode
        into quantum states.

    series_indicies : list[list[int]]
        A list of sequences, where each inner list contains row indices
        corresponding to ordered observations in `data`.

        Example:
            [[0,1,2], [5,6,7], [10,11,12]]

        Each inner list defines one reservoir trajectory.

    encoding_strategy : encoding_strategy
        Object responsible for converting classical data rows into
        quantum encoding circuits.

    ansatz : QuantumCircuit
        Quantum circuit applied after each encoding step.

        The ansatz acts as the reservoir dynamics layer and is repeatedly
        applied between encoded observations.

        The total number of qubits in the ansatz must equal:
            encoding_strategy.n_qubits + result_getter.num_qubits

    result_getter : result_getter
        Object responsible for executing or evaluating the generated
        reservoir circuits.

        Required attributes/methods:
            - num_qubits
            - get_results(reservoir_pubs)

    Returns
    -------
    result
        A pd.Dataframe of the extracted features for each sequence, where each row corresponds to the respective
        sequence in sequence_indices.

    Notes
    -----
    Circuit construction process:
        - Encoding circuits are generated only once per unique dataframe row.
        - Each sequence produces a single quantum circuit.
        - After each encoding + ansatz application, encoding qubits are reset.
        - Reservoir qubits persist throughout the sequence, allowing temporal
          information to accumulate across time steps.

    Reservoir Design
    ----------------
    The circuit architecture resembles a quantum reservoir computer:
        - Encoding qubits act as transient input channels,
        - Reservoir qubits maintain latent quantum state information,
        - The ansatz governs state evolution and feature mixing,
        - Mid-circuit resets allow sequential data injection.

    Raises
    ------
    ValueError
        If the ansatz qubit count does not equal the combined number of
        encoding and reservoir qubits.
    """
    tot_qubits = ansatz.num_qubits
    if not tot_qubits == encoding_strategy.n_qubits + result_getter.num_qubits:
        raise ValueError(f"The number of qubits in the ansatz ({tot_qubits}) does not match the sum of encoding strategy qubits ({encoding_strategy.n_qubits}) and result getter qubits ({result_getter.num_qubits}).")
    
    encoding_qubits = list(range(tot_qubits-encoding_strategy.n_qubits, tot_qubits))
    mask = list(set(x for sublist in series_indicies for x in sublist))

    filtered_df = data.iloc[mask]
    
    encoding_pubs = encoding_strategy.generate_pubs(filtered_df)

    reservoir_pubs = []
    for series in series_indicies:
        circ = QuantumCircuit(tot_qubits)
        params = []
        
        param_prefix = 0
        for idx in series:
            pub_idx = mask.index(idx)

            enc_circ, enc_params = encoding_pubs[pub_idx]
            enc_circ = enc_circ.copy().assign_parameters({
                p: Parameter(f"p{param_prefix}_{i}") for i, p in enumerate(enc_circ.parameters)
                })
            circ.append(enc_circ, qargs=encoding_qubits)
            circ.append(ansatz, qargs=circ.qubits)
            circ.reset(encoding_qubits)

            params.extend(enc_params)
            param_prefix += 1
            
        reservoir_pubs.append((circ, params))

    result = result_getter.get_results(reservoir_pubs)
    if return_circuit:
        return result, reservoir_pubs[0][0]
    else:
        return result