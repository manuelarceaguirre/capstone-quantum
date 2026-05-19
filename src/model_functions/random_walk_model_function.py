import numpy as np


def random_walk_model_function(train_df, test_df, target_col):
    """
    Random walk forecast without drift.

    Forecasts every future value as the final observed
    value from the training set.

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
    # Last observed training value
    # ---------------------------------------------
    last_value = train_df[target_col].iloc[-1]

    # ---------------------------------------------
    # Forecast
    # ---------------------------------------------
    preds = np.repeat(last_value, len(test_df))

    dates = test_df.index

    return preds, dates