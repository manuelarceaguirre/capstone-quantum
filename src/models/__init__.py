"""Model functions and the registry the evaluation loop draws from.

Model contract
--------------
Every model is a plain function::

    def my_model_function(train_df, test_df, y_train, horizon) -> pd.Series

train_df : pd.DataFrame
    Feature rows for the training window (one track, already fit/transformed
    for this fold — never contains test information).
test_df : pd.DataFrame
    Feature rows for the test window.
y_train : pd.Series
    Horizon-specific target over the training window (indexed like train_df).
horizon : int
    Forecast horizon in months.

Returns a pd.Series of predictions indexed by ``test_df.index``.

Adding a model
--------------
1. Write the function in a module under ``src/models/`` (new file per family).
2. Import it here and add it to ``MODEL_REGISTRY`` under a short name.
3. Add that name to ``ACTIVE_MODELS`` in ``notebooks/02_evaluation_loop.ipynb``.
"""
from .baselines import (
    AR6_model_function,
    arima_model_function,
    random_walk_model_function,
)
from .konduri import (
    HAS_XGB,
    adaboost_di_model_function,
    adaboost_model_function,
    decision_tree_model_function,
    knn_di_model_function,
    random_forest_model_function,
    svr_linear_di_model_function,
    xgboost_di_model_function,
    xgboost_model_function,
)
from .readout import linear_readout_model_function

MODEL_REGISTRY = {
    "random_walk":    random_walk_model_function,
    "AR6":            AR6_model_function,
    "arima":          arima_model_function,
    "decision_tree":  decision_tree_model_function,
    "adaboost":       adaboost_model_function,
    "random_forest":  random_forest_model_function,
    "adaboost_di":    adaboost_di_model_function,
    "knn_di":         knn_di_model_function,
    "svr_linear_di":  svr_linear_di_model_function,
    "linear_readout": linear_readout_model_function,
}
if HAS_XGB:
    MODEL_REGISTRY["xgboost"] = xgboost_model_function
    MODEL_REGISTRY["xgboost_di"] = xgboost_di_model_function

__all__ = [
    "MODEL_REGISTRY",
    "AR6_model_function",
    "arima_model_function",
    "random_walk_model_function",
    "HAS_XGB",
    "adaboost_model_function",
    "decision_tree_model_function",
    "random_forest_model_function",
    "adaboost_di_model_function",
    "knn_di_model_function",
    "svr_linear_di_model_function",
    "xgboost_di_model_function",
    "xgboost_model_function",
    "linear_readout_model_function",
]
