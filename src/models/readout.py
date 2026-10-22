"""Linear readout used at the end of a reservoir computing workflow.

In quantum (and classical) reservoir computing the reservoir itself is fixed;
only a single-layer perceptron with identity activation is trained on the
reservoir features. It is fit in closed form by ridge regression, which is the
standard choice in the QRC literature. The model follows the contract in
``src/models/__init__.py`` and is intended for the reservoir feature tracks,
though it runs on any numeric track.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .konduri import _prepare_xy

# Ridge penalties searched by efficient leave-one-out CV on each training window.
READOUT_ALPHAS = np.logspace(-4, 3, 15)


def linear_readout_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, horizon: int = 0):
    """
    Single-layer perceptron readout: y_hat = w . x + b, trained by ridge regression.

    Features are imputed and standardized, the target is standardized, and the
    ridge penalty is chosen from ``READOUT_ALPHAS`` by leave-one-out CV on the
    training window only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Reservoir features for the training window.
    test_df : pd.DataFrame
        Reservoir features for the test window.
    y_train : pd.Series
        Horizon-specific training target.
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
                ("readout", RidgeCV(alphas=READOUT_ALPHAS)),
            ]
        ),
        transformer=StandardScaler(),
    )
    model.fit(X_train, y)
    preds = model.predict(X_test)
    return pd.Series(np.asarray(preds, dtype=float), index=test_df.index)
