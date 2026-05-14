from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    mean_squared_log_error,
)


def symmetric_mean_absolute_percentage_error(y_true, y_pred) -> float:
    """sMAPE with zero-denominator protection."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def mase_scale(y_train: pd.Series | np.ndarray, m: int = 1) -> float:
    """Mean absolute scaled error denominator from the training target.

    This matches the convention William documented: compute the naive one-step
    denominator fold by fold from the training target, then pool scaled test errors.
    """
    y = np.asarray(pd.Series(y_train).dropna(), dtype=float)
    if len(y) <= m:
        return float("nan")
    denom = float(np.mean(np.abs(y[m:] - y[:-m])))
    return denom if np.isfinite(denom) and denom > 1e-12 else float("nan")


def rmsse_scale(y_train: pd.Series | np.ndarray, m: int = 1) -> float:
    """Root mean squared scaled error denominator from the training target."""
    y = np.asarray(pd.Series(y_train).dropna(), dtype=float)
    if len(y) <= m:
        return float("nan")
    denom = float(np.mean((y[m:] - y[:-m]) ** 2))
    return denom if np.isfinite(denom) and denom > 1e-12 else float("nan")


def regression_metrics(
    y_true,
    y_pred,
    y_train_for_scale=None,
    seasonal_period: int = 1,
) -> dict[str, float]:
    """Return the metric set listed in Week7.md.

    Includes MSE, MAE, MAPE, sMAPE, MSLE when valid, median absolute error,
    MASE, and RMSSE. For MASE/RMSSE pass the fold's training target as
    ``y_train_for_scale``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        out["msle"] = float(mean_squared_log_error(y_true, y_pred))
    else:
        out["msle"] = float("nan")

    if y_train_for_scale is not None:
        m_scale = mase_scale(y_train_for_scale, m=seasonal_period)
        r_scale = rmsse_scale(y_train_for_scale, m=seasonal_period)
        out["mase"] = float(np.mean(np.abs(y_pred - y_true)) / m_scale) if np.isfinite(m_scale) else float("nan")
        out["rmsse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2) / r_scale)) if np.isfinite(r_scale) else float("nan")
    else:
        out["mase"] = float("nan")
        out["rmsse"] = float("nan")
    return out
