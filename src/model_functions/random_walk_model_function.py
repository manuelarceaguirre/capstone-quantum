# --- Random walk / naive persistence model ---
def random_walk_model_function(train_df, test_df, target_col):
    """
    Naive persistence forecast.

    Predicts each observation using the previous
    observed value from the target series.

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

    preds = target_col.shift(1).loc[test_df.index].values

    dates = test_df.index

    return preds, dates