# --- Random walk / naive persistence model ---
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