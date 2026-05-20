from statsmodels.tsa.ar_model import AutoReg
import numpy as np


# --- AR(6) forecasting model ---
def AR6_model_function(train_df, test_df, y_train):
    """
    Autoregressive forecasting model usin
    the previous 6 observations as lags.

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