from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:  # XGBoost is optional for teammates who do not have it installed.
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

SEED = 42


@dataclass(frozen=True)
class PredictionFrameSpec:
    """Standard output schema for Thursday's shared evaluation loop."""

    date_col: str = "date"
    y_true_col: str = "y_true"
    y_pred_col: str = "y_pred"
    model_col: str = "model"
    target_col: str = "target"
    horizon_col: str = "horizon"


def _dates(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index, name="date")
    if "date" in df.columns:
        return pd.to_datetime(df["date"])
    return pd.Series(df.index, index=df.index, name="date")


def _valid_xy(df: pd.DataFrame, target_col: str, feature_cols: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    cols = list(feature_cols)
    block = df[cols + [target_col]].copy()
    mask = block[cols].notna().all(axis=1) & block[target_col].notna()
    block = block.loc[mask]
    return block[cols], block[target_col]


def _format_predictions(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    target_name: str,
    horizon: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": _dates(test_df.loc[y_true.index]).to_numpy(),
            "target": target_name,
            "horizon": horizon,
            "model": model_name,
            "y_true": y_true.to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def predict_naive_last_value(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    model_name: str = "Naive / Random Walk",
    level_feature: str | None = None,
) -> pd.DataFrame:
    """Predict the current known level for every test row.

    If ``level_feature`` is provided, the forecast is ``test_df[level_feature]``.
    Otherwise it uses ``target_col`` shifted nowhere, which is only appropriate
    when the test dataframe already includes the current level as the target column.
    """
    del train_df, feature_cols
    source_col = level_feature or target_col
    block = test_df[[source_col, target_col]].dropna()
    return _format_predictions(
        test_df=block,
        y_true=block[target_col],
        y_pred=block[source_col].to_numpy(dtype=float),
        model_name=model_name,
        target_name=target_col,
        horizon=horizon,
    )


def predict_linear_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    model_name: str = "LinearRegression",
) -> pd.DataFrame:
    """Fit linear regression on train dataframe and return test predictions."""
    X_train, y_train = _valid_xy(train_df, target_col, feature_cols)
    X_test, y_test = _valid_xy(test_df, target_col, feature_cols)
    model = LinearRegression().fit(X_train, y_train)
    return _format_predictions(test_df, y_test, model.predict(X_test), model_name, target_col, horizon)


def _tree_models() -> dict[str, object]:
    models: dict[str, object] = {
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
            n_estimators=50,
            learning_rate=0.05,
            random_state=SEED,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    return models


def _di_models() -> dict[str, object]:
    models: dict[str, object] = {
        "AdaBoost-DI": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
            n_estimators=50,
            learning_rate=0.05,
            random_state=SEED,
        ),
        "kNN-DI": Pipeline(
            [
                ("scale", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance", metric="euclidean")),
            ]
        ),
        "SVR linear-DI": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("svr", SVR(kernel="linear", C=1.0, epsilon=0.05)),
                ]
            ),
            transformer=StandardScaler(),
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost-DI"] = XGBRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
    return models


def predict_sklearn_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    """Fit one supported sklearn-style model and return standardized predictions.

    Supported names: ``AdaBoost``, ``Random Forest``, ``AdaBoost-DI``,
    ``kNN-DI``, ``SVR linear-DI``, and ``XGBoost-DI`` if xgboost is installed.
    DI models fit StandardScaler+PCA on the training features only before the
    estimator, preserving the anti-leakage rule.
    """
    all_models = {**_tree_models(), **_di_models()}
    if model_name not in all_models:
        raise KeyError(f"Unsupported model {model_name!r}; available: {sorted(all_models)}")

    X_train, y_train = _valid_xy(train_df, target_col, feature_cols)
    X_test, y_test = _valid_xy(test_df, target_col, feature_cols)

    if model_name.endswith("-DI"):
        pca = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.80, svd_solver="full", random_state=SEED)),
            ]
        )
        X_train_fit = pca.fit_transform(X_train)
        X_test_fit = pca.transform(X_test)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    model = clone(all_models[model_name]).fit(X_train_fit, y_train)
    return _format_predictions(test_df, y_test, model.predict(X_test_fit), model_name, target_col, horizon)


def supported_model_wrappers() -> dict[str, Callable[..., pd.DataFrame]]:
    """Return model wrapper callables for David's standardized evaluator."""
    return {
        "LinearRegression": predict_linear_regression,
        "AdaBoost": lambda *args, **kwargs: predict_sklearn_model(*args, model_name="AdaBoost", **kwargs),
        "Random Forest": lambda *args, **kwargs: predict_sklearn_model(*args, model_name="Random Forest", **kwargs),
        "AdaBoost-DI": lambda *args, **kwargs: predict_sklearn_model(*args, model_name="AdaBoost-DI", **kwargs),
        "kNN-DI": lambda *args, **kwargs: predict_sklearn_model(*args, model_name="kNN-DI", **kwargs),
        "SVR linear-DI": lambda *args, **kwargs: predict_sklearn_model(*args, model_name="SVR linear-DI", **kwargs),
        "XGBoost-DI": lambda *args, **kwargs: predict_sklearn_model(*args, model_name="XGBoost-DI", **kwargs),
    }
