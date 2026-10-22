"""Signature kernel ridge regression on a feature track.

Each training date is represented by the path of the last ``SIG_WINDOW`` months of
features plus a time channel. Pairs of paths are compared with the signature
kernel, computed by solving the Goursat PDE over an RBF static kernel (the naive
solver of the ``sigkernel`` package with dyadic order 0), and kernel ridge
regression is fit on the resulting Gram matrix.

Written in NumPy: the computation needs no autograd or GPU, so there is no torch
dependency.

Follows the model contract in ``src/models/__init__.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

SIG_WINDOW = 24    # months per path
SIG_ALPHA  = 1.0   # kernel ridge penalty


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _make_paths(values, index, window):
    """One (window, D+1) path per date that has a full trailing window."""
    t = np.linspace(0.0, 1.0, window)
    paths = np.stack([
        np.column_stack([t, values[end - window + 1:end + 1]])
        for end in range(window - 1, len(values))
    ])
    return paths, index[window - 1:]


def _make_test_path(train_values, test_values, window):
    """The last window-1 training rows followed by the test row, as one path."""
    if len(test_values) != 1:
        raise ValueError(f"sig_kernel expects a 1-month test window, got {len(test_values)} rows")
    context = np.vstack([train_values[-(window - 1):], test_values])
    t = np.linspace(0.0, 1.0, window)
    return np.column_stack([t, context])[np.newaxis]   # (1, window, D+1)


# ---------------------------------------------------------------------------
# Static (RBF) kernel
# ---------------------------------------------------------------------------

def _sq_dists(x, y):
    """Pairwise squared Euclidean distances between the rows of x and y."""
    sq = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(sq, 0.0)


def _rbf_gram(X, Y, sigma):
    """
    X : (A, M, D)  Y : (B, N, D)
    Returns (A, B, M, N): the RBF kernel between every point of every pair of paths.

    Uses one matrix product over all points rather than an (A, B, M, N, D)
    broadcast, which needed ~0.5 GB and ~5 s per Gram matrix.
    """
    A, M, D = X.shape
    B, N, _ = Y.shape
    sq = _sq_dists(X.reshape(-1, D), Y.reshape(-1, D))             # (A*M, B*N)
    return np.exp(-sq / (2.0 * sigma ** 2)).reshape(A, M, B, N).transpose(0, 2, 1, 3)


def _point_sigma(paths):
    """
    RBF bandwidth by the median heuristic over individual path points, the scale
    the static kernel acts on. (Distances between whole flattened paths are ~5x
    larger and make every pair of points look nearly identical.)
    """
    pts = paths.reshape(-1, paths.shape[-1])
    sq = _sq_dists(pts, pts)
    np.fill_diagonal(sq, 0.0)
    return max(float(np.sqrt(np.median(sq[sq > 0]))), 1e-6)


# ---------------------------------------------------------------------------
# Signature kernel PDE (sigkernel naive solver, dyadic order 0)
# ---------------------------------------------------------------------------

def _sig_kernel_gram(G_static):
    """
    G_static : (A, B, M, N)  static kernel over path steps
    Returns  : (A, B)        signature kernel Gram matrix
    """
    A, B, M, N = G_static.shape

    # finite-difference increments of the static kernel
    G = (G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1]
         - G_static[:, :, 1:, :-1] - G_static[:, :, :-1, 1:])

    # boundary conditions K[:, :, 0, :] = K[:, :, :, 0] = 1
    K = np.ones((A, B, M, N), dtype=np.float64)

    for i in range(M - 1):
        for j in range(N - 1):
            inc = G[:, :, i, j]
            K[:, :, i + 1, j + 1] = (
                (K[:, :, i + 1, j] + K[:, :, i, j + 1]) * (1.0 + 0.5 * inc + inc ** 2 / 12.0)
                - K[:, :, i, j] * (1.0 - inc ** 2 / 12.0)
            )

    return K[:, :, -1, -1]


# ---------------------------------------------------------------------------
# Kernels for one fold, reused across horizons
# ---------------------------------------------------------------------------

# The Gram matrices depend only on the features, which are identical for every
# horizon of a (fold, target), so the loop's h = 1, 3, 6 calls share one computation.
_CACHE = {"key": None, "kernels": None}


def _fold_key(train_df, test_df, window):
    return (
        window,
        tuple(train_df.columns),
        train_df.index[0], train_df.index[-1], len(train_df),
        test_df.index[0], len(test_df),
        int(pd.util.hash_pandas_object(train_df, index=False).sum()),
        int(pd.util.hash_pandas_object(test_df, index=False).sum()),
    )


def _fold_kernels(train_df, test_df, window):
    """Path end dates, train Gram (n, n) and test Gram (1, n) for this fold."""
    key = _fold_key(train_df, test_df, window)
    if _CACHE["key"] != key:
        # impute and scale with training statistics only
        medians = train_df.median()
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df.fillna(medians)).astype(np.float64)
        test_scaled = scaler.transform(test_df.fillna(medians)).astype(np.float64)

        X_train, path_dates = _make_paths(train_scaled, train_df.index, window)
        X_test = _make_test_path(train_scaled, test_scaled, window)

        sigma = _point_sigma(X_train)
        K_train = _sig_kernel_gram(_rbf_gram(X_train, X_train, sigma))   # (n, n)
        K_test = _sig_kernel_gram(_rbf_gram(X_test, X_train, sigma))     # (1, n)

        _CACHE["key"] = key
        _CACHE["kernels"] = (path_dates, K_train, K_test)
    return _CACHE["kernels"]


# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------

def sig_kernel_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, horizon: int = 1):
    """
    Signature kernel ridge regression on trailing ``SIG_WINDOW``-month feature paths.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature rows for the training window.
    test_df : pd.DataFrame
        Feature rows for the test window (one month).
    y_train : pd.Series
        Horizon-specific training target, indexed like ``train_df``.
    horizon : int
        Forecast horizon. The last ``horizon - 1`` training targets end after the
        forecast origin, so they are excluded from the fit. Their feature rows are
        kept: features are known at the forecast date, and the test path needs them.

    Returns
    -------
    pd.Series with predictions indexed by test_df.index
    """
    path_dates, K_train, K_test = _fold_kernels(train_df, test_df, SIG_WINDOW)

    if horizon > 1:
        y_train = y_train.iloc[:-(horizon - 1)]
    y = y_train.reindex(path_dates)
    valid = y.notna().to_numpy()
    if valid.sum() < 12:
        raise ValueError(f"Need at least 12 training paths with a known target, got {int(valid.sum())}")
    y_fit = y.to_numpy(dtype=np.float64)[valid]

    # kernel ridge has no intercept, so fit deviations from the training mean
    y_mean = y_fit.mean()
    model = KernelRidge(alpha=SIG_ALPHA, kernel="precomputed")
    model.fit(K_train[np.ix_(valid, valid)], y_fit - y_mean)
    preds = model.predict(K_test[:, valid]) + y_mean
    return pd.Series(np.asarray(preds, dtype=float), index=test_df.index)
