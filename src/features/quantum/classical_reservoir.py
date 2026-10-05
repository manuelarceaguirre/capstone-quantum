import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List


class _SimpleReservoirRNN(nn.Module):
    """
    Fixed-weight (untrained) RNN used as a classical reservoir.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity='tanh',
            batch_first=True
        )

        # Freeze weights (reservoir = random fixed dynamics)
        for param in self.parameters():
            param.requires_grad = False


def classical_reservoir(
    data: pd.DataFrame,
    series_indicies: List[List[int]],
    num_hidden_features: int,
    return_model: bool = False
):
    """
    Classical reservoir computing using an untrained PyTorch RNN.

    Parameters
    ----------
    data : pd.DataFrame
        Input feature dataframe.

    series_indicies : List[List[int]]
        Nested list of row indices defining each sequence.

    num_hidden_features : int
        Size of reservoir hidden state.

    return_model : bool
        If True, returns trained reservoir model as well.

    Returns
    -------
    pd.DataFrame or tuple
        Hidden states per sequence (final state per sequence),
        optionally with model.
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    mask = list(set(x for sublist in series_indicies for x in sublist))
    filtered_df = data.iloc[mask].copy()

    X = filtered_df.to_numpy(dtype=np.float32)

    input_size = X.shape[1]

    # Reservoir model
    model = _SimpleReservoirRNN(input_size, num_hidden_features)

    results = []

    # -----------------------------
    # Process each sequence
    # -----------------------------
    for series in series_indicies:

        h = torch.zeros(1, 1, num_hidden_features)

        for idx in series:
            x = torch.tensor(
                X[mask.index(idx)],
                dtype=torch.float32
            ).view(1, 1, -1)

            # Forward step through reservoir
            out, h = model.rnn(x, h)

        # store final hidden state for this sequence
        results.append(h.squeeze(0).squeeze(0).detach().numpy())

    # -----------------------------
    # Build output dataframe
    # -----------------------------
    results_df = pd.DataFrame(
        results,
        columns=[f"h_{i}" for i in range(num_hidden_features)]
    )

    if return_model:
        return results_df, model
    else:
        return results_df