from pmdarima import auto_arima
import numpy as np


# --- Auto-ARIMA forecasting model ---
def arima_model_function(train_df, test_df, target_col):
    """
    Fits an Auto-ARIMA model on the training
    target series and forecasts over the test period.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    target_col : pd.Series
        Full target series aligned to dataset index.

    Returns
    -------
    preds : np.ndarray
    dates : pd.Index
    """

    # ---------------------------------------------
    # Training target aligned to train set
    # ---------------------------------------------
    y_train = target_col.loc[train_df.index]

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
    preds = model.predict(
        n_periods=len(test_df)
    )

    preds = np.array(preds)

    dates = test_df.index

    return preds, dates