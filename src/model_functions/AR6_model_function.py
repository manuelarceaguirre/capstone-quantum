import numpy as np
from statsmodels.tsa.ar_model import AutoReg


def AR6_model_function(train_df, test_df, target_col):
    """
    Autoregressive (AR) forecasting model using
    the previous 6 observations as lags.

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
    # Extract training series
    # ---------------------------------------------
    y_train = train_df[target_col]

    # ---------------------------------------------
    # Fit AR(6) model
    # ---------------------------------------------
    model = AutoReg(
        y_train,
        lags=6,
        old_names=False
    ).fit()

    # ---------------------------------------------
    # Forecast
    # ---------------------------------------------
    preds = model.predict(
        start=len(y_train),
        end=len(y_train) + len(test_df) - 1
    )

    preds = np.array(preds)

    dates = test_df.index

    return preds, dates