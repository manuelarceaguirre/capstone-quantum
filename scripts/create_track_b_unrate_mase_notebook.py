from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "03_track_b_unrate_fixed_window_mase.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells = []

def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))

def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))

md(r"""
# Notebook 03 · Track B UNRATE forecasts with 2-year fixed windows and MASE

**Request.** Rerun results using the **Track B** dataset, **UNRATE**, a **2-year fixed window**, look-ahead horizons **1, 3, and 6 months**, and **Mean Absolute Scaled Error (MASE)** as the performance metric.

This notebook is deliberately explicit about assumptions because the phrase "2-year fixed window" can mean different things in forecasting. The implementation here uses a fixed 24-month **input/history window** for every supervised example. The train/test evaluation folds remain the shared project folds so these results are still comparable to other capstone notebooks. If the intended meaning was instead a 24-month rolling **training** window, that is a different experiment and should be run separately.
""")

md(r"""
## 1 · Experiment assumptions

| Area | Assumption used in this notebook | Why / implication |
|---|---|---|
| Dataset | Use `data/processed/track_B_curated.parquet` as the feature source. | This is the project Track B dataset from `01_data_prep.ipynb`. |
| UNRATE feature | The saved Track B column `UNRATE` is a **stationary month-to-month change**, not the unemployment-rate level. This notebook renames it to `UNRATE_delta` for clarity. | Prevents confusing transformed features with the level target. |
| UNRATE level | Add `UNRATE_level` from `levels_panel.parquet` to the feature panel. | If we forecast the unemployment-rate level, the current level is known at forecast origin and is essential for a fair baseline/model comparison. |
| Target | Forecast the future **UNRATE level**: `UNRATE_level[t+h]`. | MASE is most interpretable for level forecasts. Forecasting the differenced stationary UNRATE series would make MASE less economically intuitive. |
| Horizons | `h = {1, 3, 6}` months. | Per request. |
| Fixed window | Each model input is the trailing 24 months of Track B features ending at forecast origin `t`. | This matches the project/quantum need for constant-length windows. |
| Flattening | The 24 × feature matrix is flattened into one vector for classical models. | Tree/SVR/kNN models consume tabular vectors, not 3D tensors. |
| Train/test folds | Use the existing shared expanding folds from `folds.json`. | Keeps comparability with prior notebooks. This is **not** a 24-month rolling training window. |
| Horizon leakage | For horizon `h`, remove the last `h` months of each training fold because their future labels would not yet be known. | Avoids training on unavailable future outcomes. |
| Overlapping test folds | Existing 5-year test blocks overlap. We keep the latest prediction for each `(model, horizon, date)`. | Avoids over-weighting repeated target dates. |
| Metric | Mean Absolute Scaled Error: `mean(abs(error) / scale)`, where `scale` is the in-fold mean absolute one-step change in the training UNRATE level. | MASE < 1 means the model beats the in-fold naive one-step scale on average. |
| Paper models | Use the same Konduri-inspired model set where meaningful: ARD, naive/random-walk, AdaBoost, Random Forest, AdaBoost-DI, kNN-DI, SVR linear-DI, XGBoost-DI. | These came from Konduri & Li Table 1 and the best-model discussion in Tables 2-8. |
| Hyperparameters | Fixed simple hyperparameters; no tuning on test data. | Avoids hidden leakage and makes the notebook transparent. |
""")

md(r"""
## 2 · Load data and construct Track B + UNRATE level panel
""")

code(r"""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRACK_B_PATH = DATA_DIR / "track_B_curated.parquet"
LEVELS_PATH = DATA_DIR / "levels_panel.parquet"
FOLDS_PATH = DATA_DIR / "folds.json"

track_b_raw = pd.read_parquet(TRACK_B_PATH)
levels = pd.read_parquet(LEVELS_PATH)
track_b_raw.index = pd.to_datetime(track_b_raw.index)
levels.index = pd.to_datetime(levels.index)

# Rename the transformed/stationary UNRATE feature for clarity.
track_b = track_b_raw.rename(columns={"UNRATE": "UNRATE_delta"}).copy()
track_b["UNRATE_level"] = levels["UNRATE"].reindex(track_b.index)

# Keep deterministic column order.
front = ["UNRATE_level", "UNRATE_delta"]
track_b = track_b[front + [c for c in track_b.columns if c not in front]]

print("Track B raw columns     :", list(track_b_raw.columns))
print("Track B modeling columns:", list(track_b.columns))
print("Shape:", track_b.shape)
print("Date range:", track_b.index.min(), "to", track_b.index.max())
print("Missing cells:", int(track_b.isna().sum().sum()))
track_b.head()
""")

code(r"""
with open(FOLDS_PATH) as f:
    folds_json = json.load(f)["shared"]

folds = []
for i, f in enumerate(folds_json):
    train_idx = track_b.loc[f["train_start"]:f["train_end"]].index
    test_idx = track_b.loc[f["test_start"]:f["test_end"]].index
    folds.append({**f, "fold_id": i, "train_idx": train_idx, "test_idx": test_idx})

WINDOW_MONTHS = 24
HORIZONS = [1, 3, 6]
TARGET = "UNRATE_level"
LEVEL_TARGET_COL = "UNRATE"
DEDUP_POLICY = "latest"
SEED = 42

print(f"Loaded {len(folds)} folds")
print("First fold:", folds[0]["train_start"], "→", folds[0]["train_end"], "/ test", folds[0]["test_start"], "→", folds[0]["test_end"])
""")

md(r"""
## 3 · Build fixed 24-month window features

For each forecast origin `t`, the feature vector contains all Track B columns for months `t-23` through `t`, flattened in chronological order. The target for horizon `h` is `UNRATE_level[t+h]`.
""")

code(r"""
def build_fixed_window_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    '''Flatten trailing fixed windows. Row at t contains df[t-window+1 : t].'''
    frames = []
    for lag in range(window - 1, -1, -1):
        shifted = df.shift(lag)
        suffix = f"t-{lag}" if lag else "t"
        shifted.columns = [f"{col}__{suffix}" for col in df.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)

X_window = build_fixed_window_features(track_b, WINDOW_MONTHS)
y_level = levels[LEVEL_TARGET_COL].reindex(track_b.index)

print("X_window shape:", X_window.shape)
print("Feature count =", WINDOW_MONTHS, "months ×", track_b.shape[1], "features =", X_window.shape[1])
X_window.dropna().head()
""")

md(r"""
## 4 · Models

- `ARD(6)`: linear autoregression using the latest six monthly UNRATE levels.
- `Naive / Random Walk`: predicts the current UNRATE level for every horizon.
- `AdaBoost` and `Random Forest`: use the flattened 24-month Track B window directly.
- `AdaBoost-DI`, `kNN-DI`, `SVR linear-DI`, `XGBoost-DI`: first reduce the flattened 24-month Track B window with training-fold PCA (`80%` variance), then fit the model.
""")

code(r"""
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

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print("XGBoost unavailable:", repr(e))


def full_model_factories():
    return {
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


def di_model_factories():
    models = {
        "AdaBoost-DI": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
            n_estimators=50,
            learning_rate=0.05,
            random_state=SEED,
        ),
        "kNN-DI": Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance", metric="euclidean")),
        ]),
        "SVR linear-DI": TransformedTargetRegressor(
            regressor=Pipeline([
                ("scale", StandardScaler()),
                ("svr", SVR(kernel="linear", C=1.0, epsilon=0.05)),
            ]),
            transformer=StandardScaler(),
        ),
    }
    if HAS_XGB:
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

MODEL_ORDER = ["ARD(6)", "Naive / Random Walk", "AdaBoost", "Random Forest", "AdaBoost-DI", "kNN-DI", "SVR linear-DI", "XGBoost-DI"]
print("Models:", MODEL_ORDER)
""")

code(r"""
def make_ard_features(y: pd.Series, lags: int = 6) -> pd.DataFrame:
    return pd.concat([y.shift(lag).rename(f"UNRATE_level_lag{lag}") for lag in range(lags)], axis=1)


def mase_scale(y_train_level: pd.Series) -> float:
    diffs = y_train_level.dropna().diff().abs().dropna()
    scale = float(diffs.mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid MASE scale; training series may be constant or empty.")
    return scale


def append_rows(rows, model, h, fold, dates, y_true, y_pred, feature_set, scale, extra=None):
    extra = extra or {}
    for dt, yt, yp in zip(dates, y_true, y_pred):
        abs_err = abs(float(yt) - float(yp))
        rows.append({
            "model": model,
            "target": TARGET,
            "h": h,
            "fold_id": fold["fold_id"],
            "fold_train_end": fold["train_end"],
            "date": dt,
            "y_true": float(yt),
            "y_pred": float(yp),
            "abs_error": abs_err,
            "mase_scale": scale,
            "scaled_abs_error": abs_err / scale,
            "feature_set": feature_set,
            **extra,
        })


def valid_xy(X: pd.DataFrame, y: pd.Series, idx):
    Xb = X.reindex(idx)
    yb = y.reindex(idx)
    mask = Xb.notna().all(axis=1) & yb.notna()
    good_idx = idx[mask.values]
    return good_idx, X.loc[good_idx], y.loc[good_idx]
""")

md(r"""
## 5 · Run forecasts
""")

code(r"""
def fit_predict_one(h: int, fold: dict) -> list[dict]:
    y_h = y_level.shift(-h)
    train_idx = fold["train_idx"]
    test_idx = fold["test_idx"]
    train_end = pd.Timestamp(fold["train_end"])
    supervised_train_idx = train_idx[train_idx <= train_end - pd.DateOffset(months=h)]
    test_eval_idx = y_h.reindex(test_idx).dropna().index
    scale = mase_scale(y_level.reindex(supervised_train_idx))
    rows = []

    # ARD(6) on UNRATE levels.
    X_ard = make_ard_features(y_level, lags=6)
    tr_idx, Xtr, ytr = valid_xy(X_ard, y_h, supervised_train_idx)
    te_idx, Xte, yte = valid_xy(X_ard, y_h, test_eval_idx)
    if len(tr_idx) > 10 and len(te_idx) > 0:
        model = LinearRegression()
        model.fit(Xtr, ytr)
        append_rows(rows, "ARD(6)", h, fold, te_idx, yte, model.predict(Xte), "unrate_level_lags", scale)

    # Naive / random-walk level forecast.
    naive_pred = y_level.reindex(test_eval_idx)
    valid = naive_pred.notna() & y_h.reindex(test_eval_idx).notna()
    naive_idx = test_eval_idx[valid.values]
    append_rows(rows, "Naive / Random Walk", h, fold, naive_idx, y_h.loc[naive_idx], naive_pred.loc[naive_idx], "current_unrate_level", scale)

    # Full Track B fixed-window models.
    tr_idx, Xtr, ytr = valid_xy(X_window, y_h, supervised_train_idx)
    te_idx, Xte, yte = valid_xy(X_window, y_h, test_eval_idx)
    if len(tr_idx) > 20 and len(te_idx) > 0:
        for name, estimator in full_model_factories().items():
            model = clone(estimator)
            model.fit(Xtr, ytr)
            append_rows(rows, name, h, fold, te_idx, yte, model.predict(Xte), "track_b_24m_window", scale, {"n_features": Xtr.shape[1]})

        # PCA-DI models fit PCA on training window features only.
        pca_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.80, svd_solver="full", random_state=SEED)),
        ])
        Xtr_di = pca_pipe.fit_transform(Xtr)
        Xte_di = pca_pipe.transform(Xte)
        n_di = int(pca_pipe.named_steps["pca"].n_components_)
        pca_var = float(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum())
        for name, estimator in di_model_factories().items():
            model = clone(estimator)
            model.fit(Xtr_di, ytr)
            append_rows(rows, name, h, fold, te_idx, yte, model.predict(Xte_di), "track_b_24m_window_pca", scale, {"n_di": n_di, "pca_var": pca_var})

    return rows
""")

code(r"""
start = time.time()
rows = []
for h in HORIZONS:
    for fold in folds:
        rows.extend(fit_predict_one(h, fold))
    print(f"finished h={h}, rows so far={len(rows):,}")

preds_raw = pd.DataFrame(rows)
preds_raw["date"] = pd.to_datetime(preds_raw["date"])
preds_raw.to_parquet(RESULTS_DIR / "track_b_unrate_24m_mase_predictions_raw.parquet", index=False)
print(f"Raw rows: {len(preds_raw):,}")
print(f"Elapsed: {(time.time() - start)/60:.2f} minutes")
preds_raw.head()
""")

code(r"""
# Deduplicate overlapping test windows.
preds = preds_raw.copy()
preds["fold_train_end"] = pd.to_datetime(preds["fold_train_end"])
preds = preds.sort_values(["model", "h", "date", "fold_train_end"])
if DEDUP_POLICY == "latest":
    preds = preds.drop_duplicates(["model", "h", "date"], keep="last")
elif DEDUP_POLICY == "earliest":
    preds = preds.drop_duplicates(["model", "h", "date"], keep="first")
else:
    raise ValueError(DEDUP_POLICY)

preds.to_parquet(RESULTS_DIR / "track_b_unrate_24m_mase_predictions.parquet", index=False)
print(f"Deduplicated rows: {len(preds):,}")
print(preds.groupby("model").size().reindex(MODEL_ORDER).dropna().astype(int))
""")

md(r"""
## 6 · MASE results

Lower is better. `MASE < 1` means the model's mean absolute error is smaller than the in-fold naive one-step scaling error.
""")

code(r"""
def summarize(pred_df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (h, model), block in pred_df.groupby(["h", "model"]):
        recs.append({
            "target": TARGET,
            "h": h,
            "model": model,
            "n": len(block),
            "mae": float(block["abs_error"].mean()),
            "mase": float(block["scaled_abs_error"].mean()),
        })
    return pd.DataFrame(recs).sort_values(["h", "mase"])

metrics = summarize(preds)
metrics.to_csv(RESULTS_DIR / "track_b_unrate_24m_mase_metrics.csv", index=False)

mase_table = metrics.pivot_table(index="model", columns="h", values="mase").reindex(MODEL_ORDER)
mase_table.style.format("{:.3f}").background_gradient(axis=None, cmap="RdYlGn_r")
""")

code(r"""
# Best model at each look-ahead horizon.
best = metrics.sort_values("mase").groupby("h").head(1).sort_values("h")
best.style.format({"mae": "{:.3f}", "mase": "{:.3f}"})
""")

code(r"""
# Compact ranking across horizons.
summary = (
    metrics.groupby("model")
    .agg(mean_mase=("mase", "mean"), median_mase=("mase", "median"), best_mase=("mase", "min"), cells=("mase", "size"))
    .sort_values("mean_mase")
)
summary.style.format({"mean_mase": "{:.3f}", "median_mase": "{:.3f}", "best_mase": "{:.3f}"})
""")

md(r"""
## 7 · Takeaways and grading notes

- This run uses **Track B + current UNRATE level** to forecast future UNRATE levels at 1, 3, and 6 months.
- The **2-year fixed window** is implemented as a 24-month feature/history window, not a 24-month training window.
- MASE is computed per prediction using the training-fold one-step UNRATE-level scale, then averaged.
- PCA for DI models is fit inside each fold only.
- Horizon leakage is avoided by purging the last `h` training months.
- Because test folds overlap, predictions are deduplicated before the final MASE table.
- If the class/professor expects a 24-month rolling training window instead, this notebook should be cloned and the fold generator changed rather than reusing `folds.json`.
""")

nb["cells"] = cells
nbf.write(nb, nb_path)
print(f"wrote {nb_path}")
