from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "curated_features_cpi_sp500"
TARGETS = ["CPIAUCSL", "S&P 500"]
HORIZONS = [1, 3, 6, 12]
SEED = 42

warnings.filterwarnings("ignore", category=ConvergenceWarning)


FEATURE_NOTES = {
    "CPIAUCSL": "Headline CPI inflation momentum; direct inflation persistence signal.",
    "PCEPI": "Alternative broad consumption price index; cross-checks CPI with a Fed-relevant price concept.",
    "PPICMM": "Producer-price commodity pressure; upstream cost signal before consumer-price pass-through.",
    "OILPRICEx": "Energy-price shock channel; important for headline inflation and equity discount/cost pressure.",
    "FEDFUNDS": "Monetary-policy stance; affects demand, inflation expectations, discount rates, and asset prices.",
    "T10YFFM": "Long-rate minus fed funds spread; captures term-structure expectations and policy tightness.",
    "UNRATE": "Labor-market slack; Phillips-curve style demand pressure and recession-risk proxy.",
    "PERMIT": "Forward-looking housing cycle; interest-sensitive demand and wealth channel.",
    "S&P PE ratio": "Equity valuation signal; connects earnings expectations and discount-rate pressure.",
    "VIXCLSx": "Equity-market risk/uncertainty proxy; forward-looking financial-stress signal.",
}


DOMAIN_RECOMMENDED = list(FEATURE_NOTES)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    X = pd.read_parquet(DATA_DIR / "stationary_panel.parquet").sort_index()
    y = pd.read_parquet(DATA_DIR / "targets.parquet").sort_index()
    folds = json.loads((DATA_DIR / "folds.json").read_text())["shared"]
    return X, y, folds


def rank_correlations(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    joined = X.join(y.rename("target"), how="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    return joined.drop(columns=["target"]).corrwith(joined["target"]).abs()


def rank_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    joined = X.join(y.rename("target"), how="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    values = mutual_info_regression(
        joined.drop(columns=["target"]).to_numpy(),
        joined["target"].to_numpy(),
        random_state=SEED,
    )
    return pd.Series(values, index=joined.drop(columns=["target"]).columns)


def lasso_selection_frequency(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[dict[str, object]],
) -> pd.Series:
    counts = pd.Series(0, index=X.columns, dtype=float)
    usable_folds = 0
    for fold in folds:
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        train_idx = X.index[(X.index >= train_start) & (X.index <= train_end)]
        X_train = X.loc[train_idx]
        y_train = y.reindex(train_idx)
        mask = y_train.notna() & X_train.notna().all(axis=1)
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]
        if len(X_train) < 80:
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lasso",
                    LassoCV(
                        cv=TimeSeriesSplit(n_splits=5),
                        max_iter=20000,
                        random_state=SEED,
                        n_jobs=None,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        coef = np.abs(model.named_steps["lasso"].coef_)
        counts += coef > 1e-10
        usable_folds += 1
    if usable_folds == 0:
        return counts
    return counts / usable_folds


def scaled_rank_score(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    ranks = series.rank(ascending=False, method="average")
    if len(ranks) == 1:
        return pd.Series(1.0, index=series.index)
    return 1.0 - (ranks - 1.0) / (len(ranks) - 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, targets, folds = load_inputs()

    rows: list[dict[str, object]] = []
    aggregate = pd.DataFrame(index=X.columns)
    aggregate["domain_selected"] = aggregate.index.isin(DOMAIN_RECOMMENDED).astype(int)

    for target in TARGETS:
        for horizon in HORIZONS:
            y_col = f"y_{target}_h{horizon}"
            y = targets[y_col]
            corr = rank_correlations(X, y)
            mi = rank_mutual_information(X, y)
            lasso = lasso_selection_frequency(X, y, folds)

            frame = pd.DataFrame(
                {
                    "abs_corr": corr,
                    "mutual_info": mi,
                    "lasso_selection_frequency": lasso,
                }
            ).fillna(0.0)
            frame["rank_score"] = (
                scaled_rank_score(frame["abs_corr"])
                + scaled_rank_score(frame["mutual_info"])
                + scaled_rank_score(frame["lasso_selection_frequency"])
            ) / 3.0

            for feature, values in frame.iterrows():
                rows.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "feature": feature,
                        "abs_corr": values["abs_corr"],
                        "mutual_info": values["mutual_info"],
                        "lasso_selection_frequency": values["lasso_selection_frequency"],
                        "rank_score": values["rank_score"],
                    }
                )

            aggregate[f"{target}_h{horizon}_score"] = frame["rank_score"]
            aggregate[f"{target}_h{horizon}_lasso_freq"] = frame["lasso_selection_frequency"]

    detail = pd.DataFrame(rows).sort_values(["target", "horizon", "rank_score"], ascending=[True, True, False])
    aggregate["mean_rank_score"] = aggregate.filter(like="_score").mean(axis=1)
    aggregate["max_lasso_frequency"] = aggregate.filter(like="_lasso_freq").max(axis=1)
    aggregate["selected_recommendation"] = aggregate.index.isin(DOMAIN_RECOMMENDED).astype(int)
    aggregate["description"] = [FEATURE_NOTES.get(idx, "") for idx in aggregate.index]
    aggregate = aggregate.sort_values(
        ["selected_recommendation", "mean_rank_score", "max_lasso_frequency"],
        ascending=[False, False, False],
    )

    detail.to_csv(OUT_DIR / "feature_screening_detail.csv", index=False)
    aggregate.reset_index(names="feature").to_csv(OUT_DIR / "feature_screening_summary.csv", index=False)

    selected = aggregate.loc[DOMAIN_RECOMMENDED].reset_index(names="feature")
    selected[["feature", "mean_rank_score", "max_lasso_frequency", "description"]].to_csv(
        OUT_DIR / "recommended_features.csv",
        index=False,
    )

    print("Recommended unified feature set:")
    for feature in DOMAIN_RECOMMENDED:
        print(f"- {feature}: {FEATURE_NOTES[feature]}")
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
