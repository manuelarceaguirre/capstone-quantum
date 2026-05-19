import numpy as np
from pmdarima import auto_arima


def arima_model_function(train_df, test_df, target_col):
    """
    Fits an Auto-ARIMA model on the training target series
    and forecasts over the test period.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    target_col : str
        Target column to forecast.
    horizon : int
        Forecast horizon (kept for interface compatibility).

    Returns
    -------
    preds : np.ndarray
    dates : pd.Index
    """

    # ---------------------------------------------
    # Extract target series
    # ---------------------------------------------
    y_train = train_df[target_col]

    # ---------------------------------------------
    # Fit Auto-ARIMA
    # ---------------------------------------------
    model = auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    # ---------------------------------------------
    # Forecast
    # ---------------------------------------------
    preds = model.predict(n_periods=len(test_df))

    dates = test_df.index

    return preds, dates