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
# Notebook 02 · Konduri ensemble-DI baselines on our FRED-MD data

**Task.** Recreate the strongest relevant models from Konduri & Li (2024) on our processed capstone data, with special attention to the paper's **ensemble machine-learning models using dimension reduction**:

- Random Forest - DI
- XGBoost - DI
- AdaBoost - DI
- Gradient Boost - DI

The notebook keeps the project substrate from `01_data_prep.ipynb` fixed so the resulting classical baseline is directly comparable with the later QRC/QSK notebooks.

**Important methodological choice.** Konduri & Li use a 120-month rolling estimation window, monthly forecast origins, horizons `h = {1,3,6,9,12}`, annualized targets, and diffusion indices selected by Bai-Ng criteria. Our current capstone substrate uses March-2026 FRED-MD, horizons `h = {1,3,6,12}`, and shared expanding folds. This notebook therefore has two goals:

1. implement the same model family and leakage controls on **our data**;
2. make every deviation from the paper explicit rather than hiding it in code.
""")

md(r"""
## 1 · What LiteParse extracted from the paper

I downloaded `Konduri_JMP.pdf` and parsed it locally with LiteParse:

```bash
curl -L https://www.tejakonduri.com/uploads/Konduri_JMP.pdf -o literature/Konduri_JMP.pdf
npx -y @llamaindex/liteparse parse literature/Konduri_JMP.pdf --no-ocr -q \
  -o literature/Konduri_JMP_liteparse.md
```

The parsed markdown is stored at `literature/Konduri_JMP_liteparse.md`. The cells below quote the specific paper snippets that determine this notebook's model universe and evaluation design.
""")

code(r"""
from pathlib import Path
import re

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

PAPER_MD = PROJECT_ROOT / "literature" / "Konduri_JMP_liteparse.md"
print(PAPER_MD)
assert PAPER_MD.exists(), "Run LiteParse first or keep the parsed markdown in literature/."
text = PAPER_MD.read_text()
lines = text.splitlines()

def show_lines(start, end):
    for i in range(start, end + 1):
        print(f"{i:4d}: {lines[i-1]}")
""")

code(r"""
# Forecast setup: POOS design, horizons, rolling window, lags.
show_lines(199, 217)
""")

code(r"""
# Model universe and the exact ensemble-DI models from Table 1.
show_lines(1073, 1104)
""")

code(r"""
# Dimension-reduction description.
show_lines(410, 424)
""")

md(r"""
## 2 · Preprocessing audit of `01_data_prep.ipynb`

Before modeling, I re-read the previous notebook and the generated metadata. The main preprocessing decisions are:

- FRED-MD March 2026 vintage, 126 raw variables, transformed using McCracken-Ng transformation codes.
- Dropped high-missing / late-start variables: `ACOGNO`, `ANDENOx`, `TWEXAFEGSMTHx`.
- Retained a 123-variable stationary panel from 1962-04 through 2026-02.
- Created four feature tracks; this notebook uses the full stationary panel and refits PCA inside each training fold.
- Current capstone targets are cumulative log growth from `t` to `t+h` for `INDPRO`, `PAYEMS`, `CPIAUCSL`, and `S&P 500`.

### Things to keep transparent

1. **Missing values.** The previous notebook describes interior filling as past-only, but the code also calls `bfill(limit=3)`. Back-filling can use future observations if it touches an interior missing block. I do **not** redo imputation here; I use the saved substrate, but this should be reviewed before final claims.
2. **Konduri target scaling.** Konduri annualizes growth by multiplying by `1200/h`. Our stored targets are unannualized cumulative log growth. Relative RMSPE rankings are invariant to multiplying a target by a horizon-specific constant, but absolute RMSPEs are not directly comparable to the paper.
3. **CPI target definition.** Konduri treats log CPI as I(2) and forecasts a transformed inflation-growth object. The capstone target currently forecasts CPI level log growth over `h`. That is a project choice, not an exact replication of Konduri's CPI target.
4. **Fold design.** The project folds are 20-year expanding train windows with 5-year test blocks stepping annually. These test blocks overlap. To avoid over-weighting duplicated dates, this notebook keeps the **latest valid forecast for each target date** when computing headline metrics.
5. **Horizon leakage.** For a target `y_{t+h}`, the training label for dates too close to the forecast origin would not have been observed yet in real time. This notebook purges the last `h` months of each training fold before fitting supervised models.
""")

code(r"""
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metadata = json.loads((DATA_DIR / "metadata.json").read_text())
print(json.dumps({k: metadata[k] for k in ["vintage", "stationary_range", "dropped_series", "adf_failures", "zero_positive_folds"]}, indent=2))
print("\nArtifacts:")
for name, info in metadata["artifacts"].items():
    print(f"  {name:28s} {info.get('shape', '')}  {info.get('desc', '')}")
""")

md(r"""
## 3 · Modeling design for this notebook

### Models

We fit the paper's four **ensemble machine-learning + diffusion-index** models:

| Notebook name | sklearn/xgboost implementation |
|---|---|
| `Random Forest - DI` | `RandomForestRegressor` |
| `XGBoost - DI` | `XGBRegressor` |
| `AdaBoost - DI` | `AdaBoostRegressor` with shallow-tree base learner |
| `Gradient Boost - DI` | `GradientBoostingRegressor` |

### Diffusion indices

Konduri uses diffusion indices from PCA. Here, for each fold:

1. fit `StandardScaler` and `PCA(n_components=0.80)` on the training macro panel only;
2. transform train and test months into PCA factors;
3. build six lags of those factors (`lag0` through `lag5`), matching the paper's six-lag convention;
4. fit each ensemble model separately for each target/horizon.

### Baseline

For relative RMSPE, we need a benchmark. We use:

- `ARD(6)` for `INDPRO`, `PAYEMS`, and `CPIAUCSL`, implemented as linear regression on six lags of the target's stationary series;
- random-walk-no-drift for `S&P 500`, equivalent to forecasting a zero future log return.

### Evaluation

- Headline metric: `RMSPE = sqrt(mean((prediction - actual)^2))`.
- Relative RMSPE: model RMSPE divided by baseline RMSPE for the same target and horizon.
- Recession subset: observations where `USREC_{t+h} = 1`, matching the paper's statement that the target observation belongs to a recession episode.
""")

code(r"""
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
DEDUP_POLICY = "latest"  # because 5-year test blocks overlap

stationary = pd.read_parquet(DATA_DIR / "stationary_panel.parquet")
targets = pd.read_parquet(DATA_DIR / "targets.parquet")
usrec = pd.read_parquet(DATA_DIR / "usrec.parquet")["USREC"]

stationary.index = pd.to_datetime(stationary.index)
targets.index = pd.to_datetime(targets.index)
usrec.index = pd.to_datetime(usrec.index)

with open(DATA_DIR / "folds.json") as f:
    folds_json = json.load(f)["shared"]

folds = []
for f in folds_json:
    train_idx = stationary.loc[f["train_start"]:f["train_end"]].index
    test_idx = stationary.loc[f["test_start"]:f["test_end"]].index
    folds.append({**f, "train_idx": train_idx, "test_idx": test_idx})

reg_targets = ["INDPRO", "PAYEMS", "CPIAUCSL", "S&P 500"]
horizons = [1, 3, 6, 12]

def target_col(series, h):
    return f"y_{series}_h{h}"

print(stationary.shape, stationary.index.min(), stationary.index.max())
print(targets.shape, targets.index.min(), targets.index.max())
print(f"Loaded {len(folds)} shared expanding folds")
""")

code(r"""
def make_lagged(df: pd.DataFrame, lags: int, prefix: str | None = None) -> pd.DataFrame:
    '''Return columns for lag0..lag(lags-1), where lag0 is information known at t.'''
    out = []
    for lag in range(lags):
        shifted = df.shift(lag)
        if prefix is None:
            shifted = shifted.add_suffix(f"_lag{lag}")
        else:
            shifted.columns = [f"{prefix}{c}_lag{lag}" for c in shifted.columns]
        out.append(shifted)
    return pd.concat(out, axis=1)


def rmspe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def model_factories():
    models = {
        "Random Forest - DI": RandomForestRegressor(
            n_estimators=40,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "AdaBoost - DI": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
            n_estimators=50,
            learning_rate=0.05,
            random_state=SEED,
        ),
        "Gradient Boost - DI": GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=SEED,
        ),
    }
    if HAS_XGB:
        models["XGBoost - DI"] = XGBRegressor(
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

MODEL_NAMES = list(model_factories().keys())
MODEL_NAMES
""")

code(r"""
def fit_predict_one(series: str, h: int, fold: dict) -> list[dict]:
    '''Fit baseline and ensemble-DI models for one target/horizon/fold.'''
    y_name = target_col(series, h)
    y = targets[y_name]

    train_idx = fold["train_idx"]
    test_idx = fold["test_idx"]
    train_end = pd.Timestamp(fold["train_end"])
    # Purge dates whose y_{t+h} would not be known as of train_end.
    supervised_train_idx = train_idx[train_idx <= train_end - pd.DateOffset(months=h)]

    # Baseline features: ARD(6), except S&P random walk predicts zero return.
    pred_rows = []
    y_test = y.reindex(test_idx)
    valid_test = y_test.notna()
    test_eval_idx = y_test.index[valid_test]

    if series == "S&P 500":
        for dt in test_eval_idx:
            pred_rows.append({
                "model": "Baseline ARD/RW",
                "target": series,
                "h": h,
                "fold": fold["train_end"],
                "date": dt,
                "y_true": float(y.loc[dt]),
                "y_pred": 0.0,
                "recession_target_month": int(usrec.shift(-h).reindex([dt]).iloc[0]) if dt in usrec.index else np.nan,
            })
    else:
        base_signal = stationary[[series]].copy()
        X_ard = make_lagged(base_signal, LAGS)
        train_mask = X_ard.reindex(supervised_train_idx).notna().all(axis=1) & y.reindex(supervised_train_idx).notna()
        test_mask = X_ard.reindex(test_eval_idx).notna().all(axis=1) & y.reindex(test_eval_idx).notna()
        tr_idx = supervised_train_idx[train_mask.values]
        te_idx = test_eval_idx[test_mask.values]
        if len(tr_idx) > 10 and len(te_idx) > 0:
            m = LinearRegression()
            m.fit(X_ard.loc[tr_idx], y.loc[tr_idx])
            preds = m.predict(X_ard.loc[te_idx])
            rec = usrec.shift(-h).reindex(te_idx)
            for dt, yp in zip(te_idx, preds):
                pred_rows.append({
                    "model": "Baseline ARD/RW",
                    "target": series,
                    "h": h,
                    "fold": fold["train_end"],
                    "date": dt,
                    "y_true": float(y.loc[dt]),
                    "y_pred": float(yp),
                    "recession_target_month": int(rec.loc[dt]) if pd.notna(rec.loc[dt]) else np.nan,
                })

    # Diffusion-index features: fit scaler/PCA on training macro information only.
    pca_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)),
    ])
    pca_pipe.fit(stationary.loc[train_idx])
    factors = pd.DataFrame(
        pca_pipe.transform(stationary),
        index=stationary.index,
        columns=[f"DI{i+1:02d}" for i in range(pca_pipe.named_steps["pca"].n_components_)],
    )
    X_di = make_lagged(factors, LAGS)

    train_mask = X_di.reindex(supervised_train_idx).notna().all(axis=1) & y.reindex(supervised_train_idx).notna()
    test_mask = X_di.reindex(test_eval_idx).notna().all(axis=1) & y.reindex(test_eval_idx).notna()
    tr_idx = supervised_train_idx[train_mask.values]
    te_idx = test_eval_idx[test_mask.values]
    if len(tr_idx) <= 20 or len(te_idx) == 0:
        return pred_rows

    X_train, yy_train = X_di.loc[tr_idx], y.loc[tr_idx]
    X_test, yy_test = X_di.loc[te_idx], y.loc[te_idx]
    rec = usrec.shift(-h).reindex(te_idx)

    for name, estimator in model_factories().items():
        model = clone(estimator)
        model.fit(X_train, yy_train)
        preds = model.predict(X_test)
        for dt, yp in zip(te_idx, preds):
            pred_rows.append({
                "model": name,
                "target": series,
                "h": h,
                "fold": fold["train_end"],
                "date": dt,
                "y_true": float(yy_test.loc[dt]),
                "y_pred": float(yp),
                "recession_target_month": int(rec.loc[dt]) if pd.notna(rec.loc[dt]) else np.nan,
                "n_di": int(pca_pipe.named_steps["pca"].n_components_),
                "pca_var": float(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum()),
                "n_train_supervised": int(len(tr_idx)),
            })

    return pred_rows
""")

md(r"""
## 4 · Run the pseudo-out-of-sample exercise

This can take roughly 20-30 minutes on a laptop because it fits 4 ensemble models × 4 targets × 4 horizons × 39 folds, plus baselines. The cell checkpoints after every target/horizon block, so reruns resume instead of starting over.
""")

code(r"""
import time

start = time.time()
checkpoint = RESULTS_DIR / "konduri_ensemble_di_predictions_raw.parquet"

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
# Deduplicate overlapping test windows: keep the prediction from the latest train_end/fold for each target-date-model.
preds = preds_raw.copy()
preds["fold"] = pd.to_datetime(preds["fold"])
preds = preds.sort_values(["model", "target", "h", "date", "fold"])
if DEDUP_POLICY == "latest":
    preds = preds.drop_duplicates(["model", "target", "h", "date"], keep="last")
elif DEDUP_POLICY == "earliest":
    preds = preds.drop_duplicates(["model", "target", "h", "date"], keep="first")
else:
    raise ValueError(DEDUP_POLICY)

preds.to_parquet(RESULTS_DIR / "konduri_ensemble_di_predictions.parquet", index=False)
print(f"Deduplicated prediction rows: {len(preds):,}")
print(preds.groupby(["model"]).size().sort_index())
""")

md(r"""
## 5 · Results

The table below reports relative RMSPE versus the ARD/RW baseline. Values below 1.00 beat the baseline.
""")

code(r"""
def summarize_metrics(pred_df: pd.DataFrame, recession_only: bool = False) -> pd.DataFrame:
    df = pred_df.copy()
    if recession_only:
        df = df[df["recession_target_month"] == 1]
    recs = []
    for (target, h), block in df.groupby(["target", "h"]):
        base = block[block["model"] == "Baseline ARD/RW"]
        if len(base) == 0:
            continue
        base_r = rmspe(base["y_true"], base["y_pred"])
        for model, mb in block.groupby("model"):
            r = rmspe(mb["y_true"], mb["y_pred"])
            recs.append({
                "target": target,
                "h": h,
                "model": model,
                "n": len(mb),
                "rmspe": r,
                "baseline_rmspe": base_r,
                "relative_rmspe": r / base_r if base_r > 0 else np.nan,
            })
    return pd.DataFrame(recs).sort_values(["target", "h", "relative_rmspe"])

metrics = summarize_metrics(preds, recession_only=False)
metrics_recession = summarize_metrics(preds, recession_only=True)
metrics.to_csv(RESULTS_DIR / "konduri_ensemble_di_metrics.csv", index=False)
metrics_recession.to_csv(RESULTS_DIR / "konduri_ensemble_di_metrics_recession.csv", index=False)

rel_table = metrics.pivot_table(index=["target", "model"], columns="h", values="relative_rmspe")
rel_table = rel_table.reindex(columns=horizons)
rel_table.style.format("{:.3f}").background_gradient(axis=None, cmap="RdYlGn_r", vmin=0.8, vmax=1.2)
""")

code(r"""
# Best model by target/horizon.
best = metrics[metrics["model"] != "Baseline ARD/RW"].sort_values("relative_rmspe").groupby(["target", "h"]).head(1)
best = best.sort_values(["target", "h"])
best[["target", "h", "model", "n", "rmspe", "baseline_rmspe", "relative_rmspe"]].style.format({
    "rmspe": "{:.5f}",
    "baseline_rmspe": "{:.5f}",
    "relative_rmspe": "{:.3f}",
})
""")

code(r"""
# Recession-month subset, if enough target-month recession observations exist.
rel_rec = metrics_recession.pivot_table(index=["target", "model"], columns="h", values="relative_rmspe")
rel_rec = rel_rec.reindex(columns=horizons)
rel_rec.style.format("{:.3f}").background_gradient(axis=None, cmap="RdYlGn_r", vmin=0.7, vmax=1.3)
""")

code(r"""
# Compact comparison: how often each model beats the baseline.
beat_summary = metrics[metrics["model"] != "Baseline ARD/RW"].assign(beats_baseline=lambda d: d["relative_rmspe"] < 1)
beat_summary.groupby("model").agg(
    cells=("relative_rmspe", "size"),
    beats=("beats_baseline", "sum"),
    mean_relative_rmspe=("relative_rmspe", "mean"),
    median_relative_rmspe=("relative_rmspe", "median"),
).sort_values(["beats", "median_relative_rmspe"], ascending=[False, True]).style.format({
    "mean_relative_rmspe": "{:.3f}",
    "median_relative_rmspe": "{:.3f}",
})
""")

md(r"""
## 6 · Notes for the capstone write-up

1. **What we recreated.** The notebook implements the paper's ensemble machine-learning models using diffusion-index PCA features: Random Forest-DI, XGBoost-DI, AdaBoost-DI, and Gradient Boost-DI.
2. **What is not exact replication.** We use the project's March-2026 processed data, project horizons `{1,3,6,12}`, project target definitions, and project folds, rather than Konduri's 1960-2019 rolling 120-month window with `{1,3,6,9,12}`.
3. **Leakage controls added here.** PCA is refit inside every fold; supervised training rows whose `t+h` target would not be known at the forecast origin are purged; overlapping test blocks are deduplicated before headline metrics.
4. **Preprocessing issue to revisit.** The earlier notebook's `bfill(limit=3)` step may use future values for missing data if any back-filled gaps are interior. Before final submission, either remove it, prove it only affects leading rows that are later dropped, or document the exact affected cells.
5. **Next modeling step.** If we want closer Konduri replication, add an alternate fold generator with 120-month rolling windows and monthly forecast origins, and rebuild the targets using the paper's annualized transformation formulas.
""")

nb["cells"] = cells
nbf.write(nb, nb_path)
print(f"wrote {nb_path}")
