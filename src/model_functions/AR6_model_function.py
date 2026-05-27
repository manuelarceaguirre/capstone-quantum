import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.ar_model import AutoReg


def AR6_model_function(train_df, test_df, y_train, horizon):
    """
    Fits an AR(6) model using statsmodels AutoReg and forecasts horizon steps ahead.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training features (unused, kept for interface consistency).
    test_df : pd.DataFrame
        Test features (used for index alignment).
    y_train : pd.Series
        Training target series.
    horizon : int
        Forecast horizon.

    Returns
    -------
    pd.Series
        Forecasts indexed by test_df.index
    """
    # --- preprocessing (same as your ARIMA version) ---
    y = y_train.shift(horizon - 1).dropna()
    y.index = pd.to_datetime(y.index) 
    y = y.asfreq('MS')

    # --- fit AR(6) model ---
    model = AutoReg(y, lags=6, old_names=False).fit()

    # --- direct multi-step forecast ---
    preds = model.model.predict(model.params, start=len(y), end=len(train_df) + len(test_df))

    return pd.Series(preds, index=test_df.index).iloc[:len(test_df)]