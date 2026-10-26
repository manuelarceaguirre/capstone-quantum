"""Recurrent sequence models (vanilla RNN, GRU and LSTM) on a feature track.

Adapted from William's RNN/GRU benchmark. Follows the model contract in
``src/models/__init__.py``. Each fit sees sequences of the last ``SEQ_LEN``
months of track features and predicts the horizon-specific target at the
final month of the sequence.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .konduri import _numeric_frame

try:  # Optional; the registry skips these models if torch is unavailable.
    import torch
    from torch import nn

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    HAS_TORCH = False

SEQ_LEN       = 24     # months of features per input sequence
HIDDEN_SIZE   = 24
BATCH_SIZE    = 32
EPOCHS        = 30
PATIENCE      = 6      # epochs without validation improvement before stopping
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
VAL_FRACTION  = 0.2    # last share of training sequences used for early stopping
MIN_SEQUENCES = 12
SEED          = 42


def _build_net(model_name: str, input_size: int):
    """One recurrent layer followed by a linear head on the last hidden state."""
    if model_name == "rnn":
        core = nn.RNN(input_size, HIDDEN_SIZE, batch_first=True, nonlinearity="tanh")
    elif model_name == "gru":
        core = nn.GRU(input_size, HIDDEN_SIZE, batch_first=True)
    elif model_name == "lstm":
        core = nn.LSTM(input_size, HIDDEN_SIZE, batch_first=True)
    else:
        raise ValueError(f"Unsupported recurrent model: {model_name}")

    class RecurrentRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.head = nn.Linear(HIDDEN_SIZE, 1)

        def forward(self, x):
            output, _ = self.core(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    return RecurrentRegressor()


def _windows(X: np.ndarray, end_positions) -> np.ndarray:
    """Stack the SEQ_LEN rows of X ending at each position into (n, SEQ_LEN, features)."""
    return np.stack([X[p - SEQ_LEN + 1 : p + 1] for p in end_positions]).astype(np.float32)


def _fit_predict_recurrent(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           y_train: pd.Series, horizon: int) -> pd.Series:
    if not HAS_TORCH:
        raise ImportError("torch is not installed")

    X_train = _numeric_frame(train_df)
    X_test = _numeric_frame(test_df).reindex(columns=X_train.columns)
    if X_train.shape[1] == 0:
        raise ValueError("No numeric feature columns available")
    y = pd.Series(y_train).reindex(X_train.index).to_numpy(dtype=float)
    n_train = len(X_train)

    # Features are known at every training date, so impute/scale on all of them and
    # let test sequences reach back into the training window for context.
    features = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    X_all = np.vstack([features.fit_transform(X_train), features.transform(X_test)])

    # The last horizon - 1 training targets end after the forecast origin, so no
    # training sequence may end on them (look-ahead).
    ends = [p for p in range(SEQ_LEN - 1, n_train - horizon + 1) if np.isfinite(y[p])]
    if len(ends) < MIN_SEQUENCES:
        raise ValueError(f"Need at least {MIN_SEQUENCES} training sequences, got {len(ends)}")
    X_seq, y_seq = _windows(X_all, ends), y[ends]
    X_test_seq = _windows(X_all, range(n_train, n_train + len(X_test)))

    # chronological split: early stopping is judged on the latest sequences
    n_val = max(4, int(np.ceil(len(ends) * VAL_FRACTION)))
    X_fit, y_fit, X_val, y_val = X_seq[:-n_val], y_seq[:-n_val], X_seq[-n_val:], y_seq[-n_val:]

    y_mean, y_std = float(y_fit.mean()), float(y_fit.std())
    y_std = y_std if y_std > 1e-6 else 1.0
    y_fit = torch.tensor((y_fit - y_mean) / y_std, dtype=torch.float32)
    y_val = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32)
    X_fit, X_val, X_test_seq = (torch.from_numpy(a) for a in (X_fit, X_val, X_test_seq))

    torch.set_num_threads(1)  # tiny nets: threading overhead outweighs the gain
    torch.manual_seed(SEED)
    net = _build_net(model_name, X_all.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    shuffle_rng = torch.Generator().manual_seed(SEED)

    best_state, best_val, patience_left = copy.deepcopy(net.state_dict()), float("inf"), PATIENCE
    for _ in range(EPOCHS):
        net.train()
        for batch in torch.randperm(len(X_fit), generator=shuffle_rng).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = loss_fn(net(X_fit[batch]), y_fit[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(X_val), y_val))
        if val_loss < best_val:
            best_state, best_val, patience_left = copy.deepcopy(net.state_dict()), val_loss, PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        preds = net(X_test_seq).numpy() * y_std + y_mean
    return pd.Series(np.asarray(preds, dtype=float), index=test_df.index)


def rnn_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, horizon: int = 1):
    """Vanilla (tanh) RNN over the last SEQ_LEN months of track features."""
    return _fit_predict_recurrent("rnn", train_df, test_df, y_train, horizon)


def gru_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, horizon: int = 1):
    """GRU over the last SEQ_LEN months of track features."""
    return _fit_predict_recurrent("gru", train_df, test_df, y_train, horizon)


def lstm_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, horizon: int = 1):
    """LSTM over the last SEQ_LEN months of track features."""
    return _fit_predict_recurrent("lstm", train_df, test_df, y_train, horizon)
