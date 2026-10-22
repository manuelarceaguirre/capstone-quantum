"""Linear support vector regression on a feature track.

Follows the model contract in ``src/models/__init__.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from .konduri import _prepare_xy

SVM_C        = 1.0
SVM_EPSILON  = 0.1    # in standard deviations of the training target (y is standardized)
SVM_MAX_ITER = 20_000
SEED         = 42


def linear_svm_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, horizon: int = 1):
    """
    Linear SVR: y_hat = w . x + b with an epsilon-insensitive loss.

    Features are median-imputed and standardized, and the target is standardized
    so that ``SVM_EPSILON`` is measured in standard deviations of y. Without the
    target scaling, an epsilon of 0.1 is wider than nearly every monthly growth
    rate and the fit collapses to a constant.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature rows for the training window.
    test_df : pd.DataFrame
        Feature rows for the test window.
    y_train : pd.Series
        Horizon-specific training target, indexed like ``train_df``.
    horizon : int
        Forecast horizon. The last ``horizon - 1`` training targets end after the
        forecast origin, so those rows are dropped to avoid look-ahead.

    Returns
    -------
    pd.Series with predictions indexed by test_df.index
    """
    if horizon > 1:
        train_df, y_train = train_df.iloc[:-(horizon - 1)], y_train.iloc[:-(horizon - 1)]
    X_train, y, X_test = _prepare_xy(train_df, test_df, y_train)

    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("svm", LinearSVR(C=SVM_C, epsilon=SVM_EPSILON, max_iter=SVM_MAX_ITER, random_state=SEED)),
            ]
        ),
        transformer=StandardScaler(),
    )
    model.fit(X_train, y)
    preds = model.predict(X_test)
    return pd.Series(np.asarray(preds, dtype=float), index=test_df.index)
