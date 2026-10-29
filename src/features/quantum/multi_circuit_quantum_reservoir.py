from qiskit import QuantumCircuit

import pandas as pd
import numpy as np
from typing import List, Tuple

from src.features.quantum.encoding_strategy.encoding_strategy import encoding_strategy
from src.features.quantum.result_getters.result_getter import result_getter


def multi_circuit_quantum_reservoir(data : pd.DataFrame,
                      series_indicies : List[List[int]],
                      num_hidden_features : int,
                      ansatz : QuantumCircuit,
                      encoding_strategy : encoding_strategy,
                      result_getter : result_getter,
                      hidden_bounds : Tuple[float, float] = (-1.0, 1.0),
                      return_circuit : bool = False
                      ):
    """
    Runs a sequential quantum reservoir computation over multiple
    indexed time series, with one circuit execution per time step.

    At step t of a series the encoding circuit receives the features of
    row t together with the hidden state h_{t-1}, min-max rescaled from
    `hidden_bounds` to [0, 1] like the data, followed by the ansatz on all
    qubits. The result getter's output for that circuit becomes h_t. The
    hidden state starts at zero and is passed between circuits classically,
    so steps of one series run one after another, but step t of every series
    is submitted to the result getter in a single batch.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing features scaled to [0, 1].

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
        It must encode the data columns plus `num_hidden_features` hidden
        values, which come last, and should use ``scale=False`` so the hidden
        values are encoded on the same scale as the data.

    result_getter : result_getter
        Object used to execute circuits and return reservoir outputs.

    hidden_bounds : tuple of float, default=(-1.0, 1.0)
        Range of the result getter's outputs, mapped to [0, 1] before encoding:
        (-1, 1) for expectation values, (0, 1) for probabilities.

    return_circuit : bool, default=False
        If True, also returns the circuit of a single reservoir step.

    Returns
    -------
    pd.DataFrame or tuple
        Reservoir output dataframe with one row per sequence (the final hidden
        state), optionally with the circuit of one step.
    """

    tot_qubits = ansatz.num_qubits
    lo, hi = hidden_bounds
    hidden_columns = [f'h_{i}' for i in range(num_hidden_features)]

    # Hidden state of every series
    h = np.zeros((len(series_indicies), num_hidden_features))

    # Encoding + ansatz circuit for each distinct encoding circuit, built once
    step_circuits = {}

    circ = None
    for step in range(max(len(series) for series in series_indicies)):

        # Series that still have a row at this step
        active = [s for s, series in enumerate(series_indicies) if step < len(series)]

        # Data rows of this step followed by the rescaled previous hidden state
        rows = data.iloc[[series_indicies[s][step] for s in active]].reset_index(drop=True)
        hidden = pd.DataFrame(np.clip((h[active] - lo) / (hi - lo), 0, 1), columns=hidden_columns)
        encoding_pubs = encoding_strategy.generate_pubs(pd.concat([rows, hidden], axis=1))

        step_pubs = []
        for enc_circ, enc_params in encoding_pubs:
            if id(enc_circ) not in step_circuits:
                circ = QuantumCircuit(tot_qubits)
                circ.append(enc_circ, qargs=circ.qubits)
                circ.append(ansatz, qargs=circ.qubits)
                step_circuits[id(enc_circ)] = (enc_circ, circ)  # keep enc_circ alive so its id stays unique
            step_pubs.append((step_circuits[id(enc_circ)][1], enc_params))
        circ = step_pubs[-1][0]

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
