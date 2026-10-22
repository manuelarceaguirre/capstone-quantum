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
    indexed time series, with one circuit execution per time step.

    At step t of a series the encoding circuit receives the features of
    row t together with the hidden state h_{t-1}, rescaled to pi * (h + 1),
    followed by the ansatz on all qubits. The result getter's output for that
    circuit becomes h_t. The hidden state is passed between circuits
    classically, so steps of one series run one after another, but step t of
    every series is submitted to the result getter in a single batch.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing encoded features.

    series_indicies : List[List[int]]
        Nested list of row indices defining each sequence. Sequences may have
        different lengths.

    num_hidden_features : int
        Number of hidden reservoir features. Must equal the number of columns
        returned by `result_getter`.

    ansatz : QuantumCircuit
        Reservoir ansatz circuit appended after encoding.

    encoding_strategy : encoding_strategy
        Object responsible for generating encoding circuits and parameters.
        Its circuit must encode the data columns plus `num_hidden_features`
        hidden values, which take the last parameters.

    result_getter : result_getter
        Object used to execute circuits and return reservoir outputs.

    return_circuit : bool, default=False
        If True, also returns the circuit of a single reservoir step.

    Returns
    -------
    pd.DataFrame or tuple
        Reservoir output dataframe with one row per sequence (the final hidden
        state), optionally with the circuit of one step.
    """

    tot_qubits = ansatz.num_qubits

    # Unique indices used across all sequences, and their position in the pubs.
    mask = sorted(set(x for sublist in series_indicies for x in sublist))
    position = {idx: i for i, idx in enumerate(mask)}

    # Hidden feature columns are placeholders; their values are set per step.
    filtered_df = data.iloc[mask].copy()
    for i in range(num_hidden_features):
        filtered_df[f'h_{i}'] = 0.0

    # Precompute encoding circuits and parameters
    encoding_pubs = encoding_strategy.generate_pubs(filtered_df)

    # Hidden state of every series
    h = np.zeros((len(series_indicies), num_hidden_features))

    circ = None
    for step in range(max(len(series) for series in series_indicies)):

        # Series that still have a row at this step
        active = [s for s, series in enumerate(series_indicies) if step < len(series)]

        step_pubs = []
        for s in active:
            enc_circ, enc_params = encoding_pubs[position[series_indicies[s][step]]]

            # Encoding followed by the reservoir ansatz, rebuilt for every step
            circ = QuantumCircuit(tot_qubits)
            circ.append(enc_circ, qargs=circ.qubits)
            circ.append(ansatz, qargs=circ.qubits)

            # Replace the hidden-state parameters with the rescaled previous state
            params = np.array(enc_params, dtype=float)
            params[-num_hidden_features:] = np.pi * (h[s] + 1)

            step_pubs.append((circ, params))

        # Execute this step for all active series and update their hidden states
        step_result = result_getter.get_results(step_pubs)
        if step_result.shape[1] != num_hidden_features:
            raise ValueError(
                f"result_getter returned {step_result.shape[1]} features but "
                f"num_hidden_features={num_hidden_features}"
            )
        h[active] = step_result.to_numpy()
        columns = step_result.columns

    results_df = pd.DataFrame(h, columns=columns)

    if return_circuit:
        return results_df, circ
    else:
        return results_df
