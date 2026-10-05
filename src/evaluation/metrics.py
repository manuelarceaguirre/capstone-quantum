"""Forecast error metrics."""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)

METRIC_COLS = ["mse", "mae", "median_ae", "rmse", "mase", "rmspe"]


def naive_mae(y_train, horizon):
    """
    In-sample MAE of the h-step naive forecast y[t] = y[t-h] on the training
    target. Used as the MASE denominator.
    """
    try:
        y_naive_pred = y_train.shift(horizon).dropna()
        y_train_aligned = y_train[y_naive_pred.index]
        return mean_absolute_error(y_train_aligned, y_naive_pred)
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred, naive_mae):
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true    : array-like — actual target values from test window
    y_pred    : array-like — model predictions
    naive_mae : float — MAE of the naive baseline on the training window

    Returns
    -------
    dict of metric name -> float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae       = mean_absolute_error(y_true, y_pred)
    mse       = mean_squared_error(y_true, y_pred)
    rmse      = np.sqrt(mse)
    median_ae = median_absolute_error(y_true, y_pred)

    # avoid division by zero on flat training series
    if naive_mae == 0 or np.isnan(naive_mae):
        mase = np.nan
    else:
        mase = mae / naive_mae

    # RMSPE: Root Mean Squared Percentage Error
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage_errors = ((y_true - y_pred) / y_true) ** 2
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        rmspe = np.sqrt(np.mean(percentage_errors)) if len(percentage_errors) > 0 else np.nan

    return {
        "mse":       mse,
        "mae":       mae,
        "median_ae": median_ae,
        "rmse":      rmse,
        "mase":      mase,
        "rmspe":     rmspe,
    }
