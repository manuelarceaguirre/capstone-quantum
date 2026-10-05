"""Fixed sliding-window pseudo-out-of-sample folds."""


def generate_folds(index, window_months, test_months, step_months):
    """
    Generate fixed sliding window pseudo-OOS folds.

    Parameters
    ----------
    index         : pd.DatetimeIndex — full date index of the panel
    window_months : int — training window length in months
    test_months   : int — test window length in months
    step_months   : int — step size between folds in months

    Returns
    -------
    folds : list of dict with train/test date boundaries
    """
    folds = []
    n = len(index)

    for start in range(0, n - window_months - test_months + 1, step_months):
        train_start = index[start]
        train_end   = index[start + window_months - 1]
        test_start  = index[start + window_months]
        test_end    = index[start + window_months + test_months - 1]

        folds.append({
            "fold_id":     len(folds),
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
        })

    return folds
