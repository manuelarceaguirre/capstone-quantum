"""Univariate baseline models: random walk, AR(6), and auto-ARIMA.

All functions follow the model contract in ``src/models/__init__.py``. They use
only ``y_train`` and ignore the feature columns in ``train_df``/``test_df``.
"""
import numpy as np
import pandas as pd


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
    Direct AR(6) forecast of the h-month change.

    y_train[t] is the change from t to t+h, so at date t the latest known value is
    y_train[t-h]. The model regresses y[t] on y[t-h], ..., y[t-h-5] by OLS and
    applies the fit once at each test date. Unlike iterating a one-step AR(6)
    h times, a single large shock at the end of the window cannot compound.

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
    lags = 6
    y = np.asarray(y_train, dtype=float)
    n = len(y)
    if len(test_df) > horizon:
        raise ValueError(f"AR6 needs test windows of at most {horizon} months, got {len(test_df)}")

    # design row for date t: intercept and the 6 latest values known at t
    def design(t):
        return np.r_[1.0, y[t - horizon - np.arange(lags)]]

    # training dates whose target is known at the forecast origin (t <= n - h)
    rows = range(horizon + lags - 1, n - horizon + 1)
    X = np.array([design(t) for t in rows])
    beta = np.linalg.lstsq(X, y[list(rows)], rcond=None)[0]

    preds = [design(n + k) @ beta for k in range(len(test_df))]
    return pd.Series(preds, index=test_df.index)


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
