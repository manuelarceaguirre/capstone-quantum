from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


@dataclass(frozen=True)
class RecurrentConfig:
    sequence_length: int = 24
    hidden_size: int = 24
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    epochs: int = 30
    patience: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class RecurrentRegressor(nn.Module):
    """Small RNN/GRU forecaster matching William's benchmark settings."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        model_name: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if model_name == "rnn":
            self.core = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=recurrent_dropout,
            )
        elif model_name == "gru":
            self.core = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        else:
            raise ValueError(f"Unsupported recurrent model: {model_name}")
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.core(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.select_dtypes(include=[np.number]).copy()
    return out.loc[:, ~out.columns.duplicated()]


def _build_train_sequences(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = _numeric_frame(train_df)
    y = pd.Series(y_train, index=y_train.index, name="target").reindex(X_train.index)

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for end_pos in range(sequence_length - 1, len(X_train)):
        seq = X_train.iloc[end_pos - sequence_length + 1 : end_pos + 1]
        y_val = y.iloc[end_pos]
        if pd.isna(y_val):
            continue
        if not np.isfinite(seq.to_numpy()).all():
            continue
        sequences.append(seq.to_numpy(dtype=np.float32))
        targets.append(float(y_val))

    if len(sequences) < 12:
        raise ValueError(f"Need at least 12 training sequences, got {len(sequences)}")
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _build_test_sequences(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    train_tail = _numeric_frame(train_df).reindex(columns=columns).tail(sequence_length - 1)
    test_features = _numeric_frame(test_df).reindex(columns=columns)
    context = pd.concat([train_tail, test_features], axis=0)

    sequences: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for date in test_features.index:
        end_pos = context.index.get_loc(date)
        seq = context.iloc[end_pos - sequence_length + 1 : end_pos + 1]
        if seq.shape[0] != sequence_length:
            continue
        if not np.isfinite(seq.to_numpy()).all():
            continue
        sequences.append(seq.to_numpy(dtype=np.float32))
        dates.append(pd.Timestamp(date))

    if not sequences:
        raise ValueError("No valid test sequences could be built")
    return np.asarray(sequences, dtype=np.float32), pd.DatetimeIndex(dates)


def _split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(4, int(np.ceil(len(X) * 0.2)))
    if val_size >= len(X):
        val_size = max(1, len(X) // 3)
    split_at = len(X) - val_size
    return X[:split_at], y[:split_at], X[split_at:], y[split_at:]


def _standardize(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    feature_mean = X_fit.mean(axis=(0, 1), keepdims=True)
    feature_std = X_fit.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    target_mean = float(y_fit.mean())
    target_std = float(y_fit.std())
    if target_std < 1e-6:
        target_std = 1.0

    return (
        (X_fit - feature_mean) / feature_std,
        (y_fit - target_mean) / target_std,
        (X_val - feature_mean) / feature_std,
        (y_val - target_mean) / target_std,
        (X_test - feature_mean) / feature_std,
        target_mean,
        target_std,
    )


def _fit_predict_recurrent(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    config: RecurrentConfig,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    _set_seed(SEED)
    device = _choose_device()

    train_features = _numeric_frame(train_df)
    if train_features.shape[1] == 0:
        raise ValueError("No numeric feature columns available")

    X_train, y = _build_train_sequences(train_features, y_train, config.sequence_length)
    X_test, test_dates = _build_test_sequences(
        train_df=train_features,
        test_df=test_df,
        columns=train_features.columns.tolist(),
        sequence_length=config.sequence_length,
    )
    X_fit, y_fit, X_val, y_val = _split_train_validation(X_train, y)
    X_fit, y_fit, X_val, y_val, X_test, y_mean, y_std = _standardize(
        X_fit,
        y_fit,
        X_val,
        y_val,
        X_test,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.float32),
        ),
        batch_size=min(config.batch_size, len(X_fit)),
        shuffle=True,
    )

    model = RecurrentRegressor(
        input_size=X_train.shape[2],
        hidden_size=config.hidden_size,
        model_name=model_name,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_left = config.patience

    for _ in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_tensor), y_val_tensor).cpu().item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(X_test_tensor).cpu().numpy()

    predictions = pred_std * y_std + y_mean
    return np.asarray(predictions, dtype=float), test_dates


def rnn_model_function(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
):
    """Shared evaluator wrapper for William's vanilla RNN model."""
    return _fit_predict_recurrent("rnn", train_df, test_df, y_train, RecurrentConfig())


def gru_model_function(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
):
    """Shared evaluator wrapper for William's GRU model."""
    return _fit_predict_recurrent("gru", train_df, test_df, y_train, RecurrentConfig())
