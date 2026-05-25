from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:  # Optional; the notebook skips this model if xgboost is unavailable.
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    HAS_XGB = False

SEED = 42


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric model inputs with duplicate columns removed."""
    out = df.select_dtypes(include=[np.number]).copy()
    return out.loc[:, ~out.columns.duplicated()]


def _prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Align Jack's train/test feature frames with the fold's training target.

    Jack's latest Track-B notebook calls every model as
    ``model_fn(train_df, test_df, y_train)`` where ``y_train`` is already the
    horizon-specific target series for the training dates. The test target is
    handled by the evaluator, so the model only returns predictions and dates.
    """
    X_train = _numeric_frame(train_df)
    X_test = _numeric_frame(test_df).reindex(columns=X_train.columns)

    y = pd.Series(y_train, index=y_train.index, name="target").reindex(X_train.index)
    valid_y = y.notna()
    X_train = X_train.loc[valid_y]
    y = y.loc[valid_y].astype(float)

    if len(y) < 12:
        raise ValueError(f"Need at least 12 non-missing training targets, got {len(y)}")
    if X_train.shape[1] == 0:
        raise ValueError("No numeric feature columns available")

    return X_train, y, X_test


def _fit_predict_model(estimator, train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    X_train, y, X_test = _prepare_xy(train_df, test_df, y_train)
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", clone(estimator)),
        ]
    )
    model.fit(X_train, y)
    preds = model.predict(X_test)
    return np.asarray(preds, dtype=float), test_df.index


def adaboost_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    estimator = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
        n_estimators=50,
        learning_rate=0.05,
        random_state=SEED,
    )
    return _fit_predict_model(estimator, train_df, test_df, y_train)


def random_forest_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    estimator = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    return _fit_predict_model(estimator, train_df, test_df, y_train)


def _fit_predict_di_model(estimator, train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    X_train, y, X_test = _prepare_xy(train_df, test_df, y_train)
    n_components = 0.80 if min(X_train.shape) > 1 else 1
    diffusion_index = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full", random_state=SEED)),
        ]
    )
    X_train_di = diffusion_index.fit_transform(X_train)
    X_test_di = diffusion_index.transform(X_test)

    model = clone(estimator)
    model.fit(X_train_di, y)
    preds = model.predict(X_test_di)
    return np.asarray(preds, dtype=float), test_df.index


def adaboost_di_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    estimator = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
        n_estimators=50,
        learning_rate=0.05,
        random_state=SEED,
    )
    return _fit_predict_di_model(estimator, train_df, test_df, y_train)


def knn_di_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance", metric="euclidean")),
        ]
    )
    return _fit_predict_di_model(estimator, train_df, test_df, y_train)


def svr_linear_di_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    estimator = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("scale", StandardScaler()),
                ("svr", SVR(kernel="linear", C=1.0, epsilon=0.05)),
            ]
        ),
        transformer=StandardScaler(),
    )
    return _fit_predict_di_model(estimator, train_df, test_df, y_train)


def xgboost_di_model_function(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series):
    if not HAS_XGB or XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    estimator = XGBRegressor(
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
    return _fit_predict_di_model(estimator, train_df, test_df, y_train)
