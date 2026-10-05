"""Univariate baseline models: random walk, AR(6), and auto-ARIMA.

All functions follow the model contract in ``src/models/__init__.py``. They use
only ``y_train`` and ignore the feature columns in ``train_df``/``test_df``.
"""
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def random_walk_model_function(train_df, test_df, y_train, horizon):
    """
    Naive persistence forecast.

    Predicts each observation using the previous
    observed value from the target series.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    y_train : pd.Series
        Target series from training window.
    horizon : int
        Forecast horizon in periods.

    Returns
    -------
    pd.Series with predictions indexed by test_df.index
    """

    preds = y_train.shift(horizon-1).iloc[-1]

    return pd.Series(preds, index=test_df.index)


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
    y = y_train.shift(horizon - 1).dropna()
    y.index = pd.to_datetime(y.index)
    y = y.asfreq('MS')

    # --- fit AR(6) model ---
    model = AutoReg(y, lags=6, old_names=False).fit()

    # --- direct multi-step forecast ---
    preds = model.model.predict(model.params, start=len(y), end=len(train_df) + len(test_df))

    return pd.Series(preds, index=test_df.index).iloc[:len(test_df)]


def arima_model_function(train_df, test_df, y_train, horizon):
    """
    Fits Auto-ARIMA once on training data and forecasts horizon steps ahead.
    No recursive refitting.
    """
    from pmdarima import auto_arima  # imported lazily: slow import, only needed here

    y = y_train.shift(horizon - 1).dropna()
    y.index = pd.to_datetime(y.index)
    y = y.asfreq('MS')
    # --- single fit ---
    model=auto_arima(y,start_p=0,d=1,start_q=0,
        max_p=5,max_d=5,max_q=5, start_P=0,
        D=1, start_Q=0, max_P=5,max_D=5,
        max_Q=5, m=12, seasonal=False,
        error_action='warn',
        supress_warnings=True,
        random_state=20,n_fits=50)

    # --- direct multi-step forecast ---
    n_periods=(horizon+len(test_df))
    prediction = pd.DataFrame(model.predict(n_periods = n_periods),index=test_df.index)

    return prediction.iloc[-len(test_df):, 0]
