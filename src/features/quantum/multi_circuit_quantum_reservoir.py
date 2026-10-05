from qiskit import QuantumCircuit

import pandas as pd
import numpy as np
from typing import List

from src.features.quantum.encoding_strategy.encoding_strategy import encoding_strategy
from src.features.quantum.result_getters.result_getter import result_getter


def multi_circuit_quantum_reservoir(data : pd.DataFrame,
                      series_indicies : List[List[int]],
                      num_hidden_features : int,
                      ansatz : QuantumCircuit,
                      encoding_strategy : encoding_strategy,
                      result_getter : result_getter,
                      return_circuit : bool = False
                      ):
    """
    Runs a sequential quantum reservoir computation over multiple
    indexed time series.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing encoded features.

    series_indicies : List[List[int]]
        Nested list of row indices defining each sequence.

    num_hidden_features : int
        Number of hidden reservoir features.

    ansatz : QuantumCircuit
        Reservoir ansatz circuit appended after encoding.

    encoding_strategy : encoding_strategy
        Object responsible for generating encoding circuits and parameters.

    result_getter : result_getter
        Object used to execute circuits and return reservoir outputs.

    return_circuit : bool, default=False
        If True, also returns the final constructed circuit.

    Returns
    -------
    pd.DataFrame or tuple
        Reservoir output dataframe, optionally with final circuit.
    """

    tot_qubits = ansatz.num_qubits
       
    # Unique indices used across all sequences
    mask = list(set(x for sublist in series_indicies for x in sublist))

    filtered_df = data.iloc[mask]

    # Initialize hidden feature columns
    for i in range(num_hidden_features):
        filtered_df[f'h_{i}'] = 0
    
    # Precompute encoding circuits and parameters
    encoding_pubs = encoding_strategy.generate_pubs(filtered_df)

    # Initial hidden state
    h = np.zeros(num_hidden_features)

    results_df = None
    first = True

    for series in series_indicies:

        for idx in series:
            pub_idx = mask.index(idx)

        circ = QuantumCircuit(tot_qubits)
        
        for idx in series:
            pub_idx = mask.index(idx)

            enc_circ, params = encoding_pubs[pub_idx]

            # Append encoding and reservoir circuits
            circ.append(enc_circ, qargs=circ.qubits)
            circ.append(ansatz, qargs=circ.qubits)

            # Rescale hidden state to rotation parameter range
            h_params = np.pi * (h + 1)

            # Replace final parameters with hidden state values
            params[-len(h_params):] = h_params

            # Execute circuit and update hidden state
            circ_result = result_getter.get_results([(circ, params)])
            h = circ_result.iloc[0].to_numpy()

        # initialize dataframe from first result
        if first:
            results_df = circ_result.copy()
            first = False
        else:
            results_df = pd.concat([results_df, circ_result], ignore_index=True)

    if return_circuit:
        return results_df, circ
    else:
        return results_df