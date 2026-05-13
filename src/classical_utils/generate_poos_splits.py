import numpy as np
import pandas as pd

def generate_poos_splits(
    df,
    target_col,
    sequence_length=24,
    forecast_horizon=1,
    step=1,
    return_full_df=False
):
    """
    Pseudo Out-of-Sample (POOS) generator.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series dataframe.
    target_col : str
        Target column name.
    sequence_length : int
        Window size for training.
    forecast_horizon : int
        Forecast horizon (currently supports 1-step usage in most models).
    step : int
        Step size between splits.
    return_full_df : bool
        If True, yields full feature dataframe windows.
        If False, yields only target series.
    
    Yields
    ------
    train, test : pd.DataFrame or pd.Series
    """

    n = len(df)

    start = sequence_length

    while start + forecast_horizon <= n:


        train_idx = slice(start - sequence_length, start)

        test_idx = slice(start, start + forecast_horizon)

        train_block = df.iloc[train_idx]
        test_block = df.iloc[test_idx]

        # --- mode switch ---
        if return_full_df:
            train_out = train_block
            test_out = test_block
        else:
            train_out = train_block[target_col]
            test_out = test_block[target_col]

        yield train_out, test_out

        start += step