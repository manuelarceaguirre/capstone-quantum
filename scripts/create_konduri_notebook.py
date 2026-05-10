from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "02_konduri_ensemble_di_baseline.ipynb"

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
# Notebook 02 · Konduri best classical models on our FRED-MD data

**Paper source:** Teja Konduri and Qian Li, *Forecasting Macroeconomic Variables: A Systematic Comparison of Machine Learning Methods* (July 17, 2024). PDF: <https://www.tejakonduri.com/uploads/Konduri_JMP.pdf>

**Task.** Recreate the paper's strongest practical classical model set on our processed capstone data, not just gradient/ensemble-DI models. Based on Konduri & Li's Table 1 model universe and Tables 2-8 results, this notebook runs the following eight model families:

1. **ARD(6)** benchmark
2. **Random Walk** benchmark for S&P 500 returns
3. **AdaBoost**
4. **AdaBoost-DI**
5. **kNN-DI**
6. **SVR linear-DI**
7. **Random Forest**
8. **XGBoost-DI**

**Why these eight?** Tables 2-6 show that real variables are often best forecast by AdaBoost, kNN with diffusion indices, SVR linear with diffusion indices, Random Forest, and XGBoost-DI. Tables 4-5/7-8 show that CPI and financial variables are hard for ML; the ARD or Random Walk baselines are themselves important winners. Therefore the fair capstone baseline is not only gradient boosting: it must include both econometric baselines and the recurrent top ML models from the paper.

**Important methodological note.** This notebook applies those model families to **our data substrate** from `01_data_prep.ipynb`. It is not an exact Konduri replication: our dataset is the March-2026 FRED-MD vintage, our horizons are `{1,3,6,12}`, and our shared folds are expanding 20-year train / 5-year test blocks. Every deviation is documented so the capstone write-up can be transparent.
""")

md(r"""
## 1 · Paper extraction with LiteParse

I downloaded and parsed the paper locally with LiteParse:

```bash
curl -L https://www.tejakonduri.com/uploads/Konduri_JMP.pdf -o literature/Konduri_JMP.pdf
npx -y @llamaindex/liteparse parse literature/Konduri_JMP.pdf --no-ocr -q \
  -o literature/Konduri_JMP_liteparse.md
```

The parsed markdown is stored at `literature/Konduri_JMP_liteparse.md`. The cells below quote the table/model lines used to choose the eight models in this notebook.
""")

code(r"""
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

PAPER_URL = "https://www.tejakonduri.com/uploads/Konduri_JMP.pdf"
PAPER_MD = PROJECT_ROOT / "literature" / "Konduri_JMP_liteparse.md"
print("Paper URL:", PAPER_URL)
print("LiteParse markdown:", PAPER_MD)
assert PAPER_MD.exists(), "Expected parsed paper markdown in literature/."

paper_text = PAPER_MD.read_text()
paper_lines = paper_text.splitlines()

def show_lines(start, end):
    for i in range(start, end + 1):
        print(f"{i:4d}: {paper_lines[i-1]}")
""")

code(r"""
# Forecast design from section 2.3: rolling POOS, horizons, six lags.
show_lines(199, 217)
""")

code(r"""
# Table 1: full model universe. This is where ARD/RW, AdaBoost, RF, kNN-DI, SVR-DI, and XGBoost-DI come from.
show_lines(1073, 1104)
""")

code(r"""
# Section 3.4: diffusion indices are PCA factors.
show_lines(410, 424)
""")

code(r"""
# Table 6: best models for real variables, including Industrial Production and Employment.
show_lines(1304, 1340)
""")

code(r"""
# Tables 7-8: CPI and S&P 500. These motivate keeping ARD/RW baselines and AdaBoost as serious contenders.
show_lines(1360, 1372)
print("\n--- S&P 500 excerpt ---")
show_lines(1416, 1428)
""")

md(r"""
## 2 · Preprocessing audit of the existing project substrate

This notebook intentionally **does not redo** `01_data_prep.ipynb`; it consumes the saved artifacts so model comparisons remain consistent across the capstone.

Key decisions from the previous notebook:

- FRED-MD March 2026 vintage.
- 126 raw variables; 123 retained after dropping `ACOGNO`, `ANDENOx`, and `TWEXAFEGSMTHx`.
- McCracken-Ng transformations applied to create a stationary panel.
- Shared targets: cumulative log growth for `INDPRO`, `PAYEMS`, `CPIAUCSL`, and `S&P 500` at `h = {1,3,6,12}`.
- Shared expanding folds: 20-year initial training window, 5-year test blocks, one-year step.

### Transparency flags

1. **Missing-value handling.** The previous notebook text says missing values are handled past-only, but the code also uses `bfill(limit=3)`. If this touched interior gaps, it can use future observations. I do not change the substrate here, but this should be audited before final claims.
2. **Konduri target scaling.** Konduri annualizes targets using `1200/h`. Our stored targets are unannualized cumulative log growth. Relative RMSPE is invariant to a horizon-specific scaling, but absolute RMSPE is not directly comparable to the paper.
3. **CPI definition.** Konduri treats log CPI as I(2). Our project target is CPI cumulative log growth over horizon `h`. This is a project target choice, not an exact CPI-target replication.
4. **Fold design.** Konduri uses 120-month rolling windows with monthly forecast origins. Our capstone substrate uses expanding windows with overlapping 5-year test blocks. The notebook deduplicates overlapping test predictions before headline metrics.
5. **Horizon leakage.** For target `y_{t+h}`, training labels in the last `h` months of a training fold would not be observed at the forecast origin. This notebook purges those rows before fitting supervised models.
""")

code(r"""
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metadata = json.loads((DATA_DIR / "metadata.json").read_text())
print(json.dumps({k: metadata[k] for k in ["vintage", "stationary_range", "dropped_series", "adf_failures", "zero_positive_folds"]}, indent=2))
print("\nArtifacts used:")
for name in ["stationary_panel.parquet", "targets.parquet", "usrec.parquet", "folds.json"]:
    print(f"  {name:26s} {metadata['artifacts'][name].get('shape', '')}  {metadata['artifacts'][name].get('desc', '')}")
""")

md(r"""
## 3 · Model design on our data

### Models and feature sets

| Model in notebook | Paper source | Feature set here | Notes |
|---|---|---|---|
| `ARD(6)` | Table 1 baseline; Tables 2-4/6-7 | six lags of target's stationary series | benchmark for non-financial targets |
| `Random Walk` | Table 1 baseline; Tables 5/8 | zero future log return | benchmark for `S&P 500` |
| `AdaBoost` | Table 1; strong in Tables 2, 3, 5, 6, 8 | full 123-variable stationary panel with six lags | tree boosting on raw macro features |
| `Random Forest` | Table 1; strong in Tables 3 and 6 | full 123-variable stationary panel with six lags | bagged trees on raw macro features |
| `AdaBoost-DI` | Table 1; recurring top DI model in Tables 2-8 | PCA diffusion indices with six lags | PCA fit per fold only |
| `kNN-DI` | Table 1; strong in Tables 2 and 6 | PCA diffusion indices with six lags | inverse-distance kNN, following the paper's kNN inverse variant |
| `SVR linear-DI` | Table 1; very strong for employment/recession horizons | PCA diffusion indices with six lags | scaled linear SVR |
| `XGBoost-DI` | Table 1; strong for employment/unemployment style variables | PCA diffusion indices with six lags | gradient boosted trees on PCA factors |

### Leakage controls

- Full-panel lag features use only information available at time `t`: `lag0` through `lag5`.
- PCA is fit on the training fold only, then applied to train/test months.
- The last `h` months of each training fold are purged before fitting because their `t+h` labels would not be known.
- Because shared 5-year test folds overlap, headline metrics keep the latest prediction for each `(model, target, horizon, date)`.
""")

code(r"""
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print("XGBoost unavailable:", repr(e))

SEED = 42
LAGS = 6
PCA_VARIANCE = 0.80
DEDUP_POLICY = "latest"

stationary = pd.read_parquet(DATA_DIR / "stationary_panel.parquet")
targets = pd.read_parquet(DATA_DIR / "targets.parquet")
usrec = pd.read_parquet(DATA_DIR / "usrec.parquet")["USREC"]

stationary.index = pd.to_datetime(stationary.index)
targets.index = pd.to_datetime(targets.index)
usrec.index = pd.to_datetime(usrec.index)

with open(DATA_DIR / "folds.json") as f:
    folds_json = json.load(f)["shared"]

folds = []
for i, f in enumerate(folds_json):
    train_idx = stationary.loc[f["train_start"]:f["train_end"]].index
    test_idx = stationary.loc[f["test_start"]:f["test_end"]].index
    folds.append({**f, "fold_id": i, "train_idx": train_idx, "test_idx": test_idx})

reg_targets = ["INDPRO", "PAYEMS", "CPIAUCSL", "S&P 500"]
horizons = [1, 3, 6, 12]

def target_col(series, h):
    return f"y_{series}_h{h}"

print("stationary:", stationary.shape, stationary.index.min(), "to", stationary.index.max())
print("targets   :", targets.shape, targets.index.min(), "to", targets.index.max())
print(f"folds     : {len(folds)}")
""")

code(r"""
def make_lagged(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    '''Return lag0..lag(lags-1), where lag0 is information known at forecast origin t.'''
    return pd.concat([df.shift(lag).add_suffix(f"_lag{lag}") for lag in range(lags)], axis=1)


def rmspe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def full_model_factories():
    '''Models selected from Konduri tables that use the full macro panel here.'''
    return {
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
            n_estimators=50,
            learning_rate=0.05,
            random_state=SEED,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=40,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def di_model_factories():
    '''Models selected from Konduri tables that use PCA diffusion-index features here.'''
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
            n_estimators=50,
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

MODEL_ORDER = ["ARD(6)", "Random Walk", "AdaBoost", "Random Forest", "AdaBoost-DI", "kNN-DI", "SVR linear-DI", "XGBoost-DI"]
print("Full-panel models:", list(full_model_factories()))
print("DI models        :", list(di_model_factories()))
print("Eight requested model families:", MODEL_ORDER)
""")

md(r"""
## 4 · Precompute leakage-safe fold features

This step caches the feature matrices so the modeling loop does not refit PCA repeatedly for every target and horizon.
""")

code(r"""
X_full_lagged = make_lagged(stationary, LAGS)

fold_feature_cache = {}
for fold in folds:
    fold_id = fold["fold_id"]
    pca_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)),
    ])
    pca_pipe.fit(stationary.loc[fold["train_idx"]])
    factors = pd.DataFrame(
        pca_pipe.transform(stationary),
        index=stationary.index,
        columns=[f"DI{i+1:02d}" for i in range(pca_pipe.named_steps["pca"].n_components_)],
    )
    fold_feature_cache[fold_id] = {
        "X_di_lagged": make_lagged(factors, LAGS),
        "n_di": int(pca_pipe.named_steps["pca"].n_components_),
        "pca_var": float(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum()),
    }

pd.DataFrame([
    {"fold_id": k, "n_di": v["n_di"], "pca_var": v["pca_var"]}
    for k, v in fold_feature_cache.items()
]).describe().round(3)
""")

code(r"""
def append_prediction_rows(rows, model_name, series, h, fold, dates, y_true, y_pred, feature_set, extra=None):
    rec = usrec.shift(-h).reindex(dates)
    extra = extra or {}
    for dt, yt, yp in zip(dates, y_true, y_pred):
        rows.append({
            "model": model_name,
            "target": series,
            "h": h,
            "fold_id": fold["fold_id"],
            "fold_train_end": fold["train_end"],
            "date": dt,
            "y_true": float(yt),
            "y_pred": float(yp),
            "feature_set": feature_set,
            "recession_target_month": int(rec.loc[dt]) if pd.notna(rec.loc[dt]) else np.nan,
            **extra,
        })


def valid_xy(X, y, idx):
    Xb = X.reindex(idx)
    yb = y.reindex(idx)
    mask = Xb.notna().all(axis=1) & yb.notna()
    good_idx = idx[mask.values]
    return good_idx, X.loc[good_idx], y.loc[good_idx]


def fit_predict_one(series: str, h: int, fold: dict) -> list[dict]:
    y = targets[target_col(series, h)]
    train_idx = fold["train_idx"]
    test_idx = fold["test_idx"]
    train_end = pd.Timestamp(fold["train_end"])
    supervised_train_idx = train_idx[train_idx <= train_end - pd.DateOffset(months=h)]
    test_eval_idx = y.reindex(test_idx).dropna().index
    rows = []

    # 1) ARD benchmark for non-financial targets.
    if series != "S&P 500":
        X_ard = make_lagged(stationary[[series]], LAGS)
        tr_idx, Xtr, ytr = valid_xy(X_ard, y, supervised_train_idx)
        te_idx, Xte, yte = valid_xy(X_ard, y, test_eval_idx)
        if len(tr_idx) > 10 and len(te_idx) > 0:
            model = LinearRegression()
            model.fit(Xtr, ytr)
            append_prediction_rows(rows, "ARD(6)", series, h, fold, te_idx, yte, model.predict(Xte), "target_lags")

    # 2) Random walk benchmark for S&P 500 log returns.
    if series == "S&P 500":
        y_test = y.reindex(test_eval_idx).dropna()
        append_prediction_rows(rows, "Random Walk", series, h, fold, y_test.index, y_test, np.zeros(len(y_test)), "rw_zero_return")

    # 3) Full-panel ML models.
    tr_idx, Xtr, ytr = valid_xy(X_full_lagged, y, supervised_train_idx)
    te_idx, Xte, yte = valid_xy(X_full_lagged, y, test_eval_idx)
    if len(tr_idx) > 20 and len(te_idx) > 0:
        for name, estimator in full_model_factories().items():
            model = clone(estimator)
            model.fit(Xtr, ytr)
            append_prediction_rows(
                rows, name, series, h, fold, te_idx, yte, model.predict(Xte), "full_panel_lags",
                {"n_train_supervised": int(len(tr_idx)), "n_features": int(Xtr.shape[1])},
            )

    # 4) PCA diffusion-index ML models.
    cache = fold_feature_cache[fold["fold_id"]]
    X_di = cache["X_di_lagged"]
    tr_idx, Xtr, ytr = valid_xy(X_di, y, supervised_train_idx)
    te_idx, Xte, yte = valid_xy(X_di, y, test_eval_idx)
    if len(tr_idx) > 20 and len(te_idx) > 0:
        for name, estimator in di_model_factories().items():
            model = clone(estimator)
            model.fit(Xtr, ytr)
            append_prediction_rows(
                rows, name, series, h, fold, te_idx, yte, model.predict(Xte), "di_pca_lags",
                {
                    "n_train_supervised": int(len(tr_idx)),
                    "n_features": int(Xtr.shape[1]),
                    "n_di": cache["n_di"],
                    "pca_var": cache["pca_var"],
                },
            )

    return rows
""")

md(r"""
## 5 · Run pseudo-out-of-sample forecasts

The cell checkpoints after each target/horizon block. If interrupted, rerun the notebook and it resumes from `results/konduri_best_models_predictions_raw.parquet`.
""")

code(r"""
start = time.time()
checkpoint = RESULTS_DIR / "konduri_best_models_predictions_raw.parquet"

if checkpoint.exists():
    preds_raw = pd.read_parquet(checkpoint)
    done = set(map(tuple, preds_raw[["target", "h"]].drop_duplicates().to_numpy()))
    rows = preds_raw.to_dict("records")
    print(f"Loaded checkpoint with {len(preds_raw):,} rows and {len(done)} completed target/horizon blocks")
else:
    done = set()
    rows = []

total_blocks = len(reg_targets) * len(horizons)
for series in reg_targets:
    for h in horizons:
        if (series, h) in done:
            print(f"skip existing {series:8s} h={h:2d}")
            continue
        block_rows = []
        for fold in folds:
            block_rows.extend(fit_predict_one(series, h, fold))
        rows.extend(block_rows)
        preds_raw = pd.DataFrame(rows)
        preds_raw["date"] = pd.to_datetime(preds_raw["date"])
        preds_raw.to_parquet(checkpoint, index=False)
        done.add((series, h))
        print(f"finished {series:8s} h={h:2d}  block_rows={len(block_rows):,}  completed={len(done)}/{total_blocks}")

preds_raw = pd.DataFrame(rows)
preds_raw["date"] = pd.to_datetime(preds_raw["date"])
preds_raw.to_parquet(checkpoint, index=False)
print(f"Raw prediction rows: {len(preds_raw):,}")
print(f"Elapsed this run: {(time.time() - start)/60:.1f} minutes")
preds_raw.head()
""")

code(r"""
# Deduplicate overlapping test windows: keep latest fold's prediction for each model/target/h/date.
preds = preds_raw.copy()
preds["fold_train_end"] = pd.to_datetime(preds["fold_train_end"])
preds = preds.sort_values(["model", "target", "h", "date", "fold_train_end"])
if DEDUP_POLICY == "latest":
    preds = preds.drop_duplicates(["model", "target", "h", "date"], keep="last")
elif DEDUP_POLICY == "earliest":
    preds = preds.drop_duplicates(["model", "target", "h", "date"], keep="first")
else:
    raise ValueError(DEDUP_POLICY)

preds.to_parquet(RESULTS_DIR / "konduri_best_models_predictions.parquet", index=False)
print(f"Deduplicated prediction rows: {len(preds):,}")
print(preds.groupby("model").size().reindex(MODEL_ORDER).dropna().astype(int))
""")

md(r"""
## 6 · Results on our data

For relative RMSPE, the benchmark is `ARD(6)` for `INDPRO`, `PAYEMS`, and `CPIAUCSL`, and `Random Walk` for `S&P 500`. Values below 1.00 beat the relevant benchmark.
""")

code(r"""
def benchmark_name(target):
    return "Random Walk" if target == "S&P 500" else "ARD(6)"


def summarize_metrics(pred_df: pd.DataFrame, recession_only: bool = False) -> pd.DataFrame:
    df = pred_df.copy()
    if recession_only:
        df = df[df["recession_target_month"] == 1]
    recs = []
    for (target, h), block in df.groupby(["target", "h"]):
        bname = benchmark_name(target)
        base = block[block["model"] == bname]
        if len(base) == 0:
            continue
        base_r = rmspe(base["y_true"], base["y_pred"])
        for model, mb in block.groupby("model"):
            r = rmspe(mb["y_true"], mb["y_pred"])
            recs.append({
                "target": target,
                "h": h,
                "model": model,
                "benchmark": bname,
                "n": len(mb),
                "rmspe": r,
                "benchmark_rmspe": base_r,
                "relative_rmspe": r / base_r if base_r > 0 else np.nan,
            })
    return pd.DataFrame(recs).sort_values(["target", "h", "relative_rmspe"])

metrics = summarize_metrics(preds, recession_only=False)
metrics_recession = summarize_metrics(preds, recession_only=True)
metrics.to_csv(RESULTS_DIR / "konduri_best_models_metrics.csv", index=False)
metrics_recession.to_csv(RESULTS_DIR / "konduri_best_models_metrics_recession.csv", index=False)

rel_table = metrics.pivot_table(index=["target", "model"], columns="h", values="relative_rmspe")
rel_table = rel_table.reindex(columns=horizons)
rel_table.style.format("{:.3f}").background_gradient(axis=None, cmap="RdYlGn_r", vmin=0.7, vmax=1.3)
""")

code(r"""
# Best non-benchmark model by target/horizon.
benchmarks = {"ARD(6)", "Random Walk"}
best = metrics[~metrics["model"].isin(benchmarks)].sort_values("relative_rmspe").groupby(["target", "h"]).head(1)
best = best.sort_values(["target", "h"])
best[["target", "h", "model", "benchmark", "n", "rmspe", "benchmark_rmspe", "relative_rmspe"]].style.format({
    "rmspe": "{:.5f}",
    "benchmark_rmspe": "{:.5f}",
    "relative_rmspe": "{:.3f}",
})
""")

code(r"""
# Recession-target-month subset. Interpret cautiously because some target/horizon cells have few recession observations.
rel_rec = metrics_recession.pivot_table(index=["target", "model"], columns="h", values="relative_rmspe")
rel_rec = rel_rec.reindex(columns=horizons)
rel_rec.style.format("{:.3f}").background_gradient(axis=None, cmap="RdYlGn_r", vmin=0.6, vmax=1.4)
""")

code(r"""
# Compact summary across target/horizon cells where each model is applicable.
summary = (
    metrics[~metrics["model"].isin({"ARD(6)", "Random Walk"})]
    .assign(beats_benchmark=lambda d: d["relative_rmspe"] < 1)
    .groupby("model")
    .agg(
        cells=("relative_rmspe", "size"),
        beats=("beats_benchmark", "sum"),
        mean_relative_rmspe=("relative_rmspe", "mean"),
        median_relative_rmspe=("relative_rmspe", "median"),
        best_relative_rmspe=("relative_rmspe", "min"),
    )
    .sort_values(["beats", "median_relative_rmspe"], ascending=[False, True])
)
summary.style.format({
    "mean_relative_rmspe": "{:.3f}",
    "median_relative_rmspe": "{:.3f}",
    "best_relative_rmspe": "{:.3f}",
})
""")

md(r"""
## 7 · Notes for the capstone write-up

1. **Paper link and table provenance.** The model list comes from Konduri & Li Table 1. The priority set is justified by Tables 2-6 for Industrial Production, Employment, and other real variables; Table 7 for CPI/nominal variables; and Table 8 for S&P 500/financial variables. Paper URL: <https://www.tejakonduri.com/uploads/Konduri_JMP.pdf>.
2. **What we implemented.** ARD(6), Random Walk, AdaBoost, Random Forest, AdaBoost-DI, kNN-DI, SVR linear-DI, and XGBoost-DI.
3. **What is not exact replication.** We use our March-2026 processed data, our project target definitions, horizons `{1,3,6,12}`, and our expanding folds rather than Konduri's 120-month rolling monthly POOS and horizons `{1,3,6,9,12}`.
4. **Leakage controls.** PCA is refit inside each fold; training rows whose `t+h` label would not yet be known are purged; overlapping test-window predictions are deduplicated before headline metrics.
5. **Preprocessing caveat.** The prior notebook's `bfill(limit=3)` should be audited. If it affects interior gaps, it introduces future information into the feature substrate.
6. **Next step for closer replication.** Add a second evaluation mode with Konduri-style 120-month rolling windows and annualized target transformations, then compare whether model rankings are stable.
""")

nb["cells"] = cells
nbf.write(nb, nb_path)
print(f"wrote {nb_path}")
