from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUT_DIR = ROOT / "results" / "classical_benchmark"
SEED = 42

REG_TARGETS = ["INDPRO", "PAYEMS", "CPIAUCSL", "S&P 500"]
CLS_TARGETS = ["USREC"]
ALL_HORIZONS = [1, 3, 6, 12]
TRACK_ORDER = ["A", "B", "C", "D", "D-mini"]


@dataclass(frozen=True)
class TrackSpec:
    name: str
    source_file: str
    needs_transform: bool = False


TRACKS = {
    "A": TrackSpec("A", "track_A_full.parquet"),
    "B": TrackSpec("B", "track_B_curated.parquet"),
    "C": TrackSpec("C", "stationary_panel.parquet", needs_transform=True),
    "D": TrackSpec("D", "stationary_panel.parquet", needs_transform=True),
    "D-mini": TrackSpec("D-mini", "stationary_panel.parquet", needs_transform=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible classical benchmark on the BMO capstone folds."
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=TRACK_ORDER,
        choices=TRACK_ORDER,
        help="Feature tracks to evaluate.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=ALL_HORIZONS,
        choices=ALL_HORIZONS,
        help="Forecast horizons to evaluate.",
    )
    parser.add_argument(
        "--include-trees",
        action="store_true",
        help="Also run random-forest baselines on Track B and D-mini.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for benchmark outputs.",
    )
    return parser.parse_args()


def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[dict[str, object]], dict]:
    tracks = {
        name: pd.read_parquet(DATA_DIR / spec.source_file).sort_index()
        for name, spec in TRACKS.items()
    }
    targets = pd.read_parquet(DATA_DIR / "targets.parquet").sort_index()
    folds = json.loads((DATA_DIR / "folds.json").read_text())["shared"]
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    return tracks, targets, folds, metadata


def make_track_transform(track_name: str) -> Pipeline | None:
    if track_name == "C":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "select",
                    SelectFromModel(
                        LassoCV(
                            cv=TimeSeriesSplit(n_splits=5),
                            max_iter=20000,
                            random_state=SEED,
                        )
                    ),
                ),
            ]
        )
    if track_name == "D":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.80, random_state=SEED)),
            ]
        )
    if track_name == "D-mini":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=15, random_state=SEED)),
            ]
        )
    return None


def regression_models(include_trees: bool) -> dict[str, object]:
    models: dict[str, object] = {
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]
        )
    }
    if include_trees:
        models["rf"] = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
    return models


def classification_models(include_trees: bool) -> dict[str, object]:
    models: dict[str, object] = {
        "logit": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    }
    if include_trees:
        models["rf"] = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )
    return models


def rows_between(index: pd.Index, start: str, end: str) -> pd.Index:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return index[(index >= start_ts) & (index <= end_ts)]


def prepare_split(
    features: pd.DataFrame,
    target: pd.Series,
    fold: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_idx = rows_between(features.index, fold["train_start"], fold["train_end"])
    test_idx = rows_between(features.index, fold["test_start"], fold["test_end"])

    y_aligned = target.reindex(features.index)
    train = features.loc[train_idx].copy()
    test = features.loc[test_idx].copy()
    y_train = y_aligned.loc[train_idx].copy()
    y_test = y_aligned.loc[test_idx].copy()

    train_mask = y_train.notna() & train.notna().all(axis=1)
    test_mask = y_test.notna() & test.notna().all(axis=1)
    return train.loc[train_mask], y_train.loc[train_mask], test.loc[test_mask], y_test.loc[test_mask]


def transformed_feature_count(transformer: Pipeline | None, X: pd.DataFrame) -> int:
    if transformer is None:
        return X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return transformer.transform(X).shape[1]


def safe_fit_transform(
    transformer: Pipeline | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if transformer is None:
        return X_train.values, X_test.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        transformer.fit(X_train, y_train)
        return transformer.transform(X_train), transformer.transform(X_test)


def evaluate_regression(
    track_name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    folds: list[dict[str, object]],
    include_trees: bool,
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    models = regression_models(include_trees)
    if include_trees is False:
        model_track_allow = {name: set(TRACK_ORDER) for name in models}
    else:
        model_track_allow = {
            "ridge": set(TRACK_ORDER),
            "rf": {"B", "D-mini"},
        }

    for model_name, model in models.items():
        if track_name not in model_track_allow[model_name]:
            continue
        for target_name in REG_TARGETS:
            for horizon in horizons:
                col = f"y_{target_name}_h{horizon}"
                if col not in targets.columns:
                    continue
                pooled_true: list[float] = []
                pooled_pred: list[float] = []
                used_folds = 0
                feature_count = None
                for fold_id, fold in enumerate(folds):
                    X_train, y_train, X_test, y_test = prepare_split(features, targets[col], fold)
                    if X_train.empty or X_test.empty:
                        continue

                    transform = make_track_transform(track_name)
                    X_train_fit, X_test_fit = safe_fit_transform(transform, X_train, y_train, X_test)

                    if feature_count is None:
                        feature_count = int(X_train_fit.shape[1])

                    fitted = clone(model)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        fitted.fit(X_train_fit, y_train)
                        y_pred = fitted.predict(X_test_fit)
                    if not np.isfinite(y_pred).all():
                        continue
                    pooled_true.extend(y_test.tolist())
                    pooled_pred.extend(np.asarray(y_pred).tolist())
                    used_folds += 1

                    for dt, truth, pred in zip(X_test.index, y_test, y_pred):
                        predictions.append(
                            {
                                "task": "regression",
                                "track": track_name,
                                "model": model_name,
                                "target": target_name,
                                "horizon": horizon,
                                "fold": fold_id,
                                "date": str(pd.Timestamp(dt).date()),
                                "y_true": float(truth),
                                "y_pred": float(pred),
                            }
                        )

                if not pooled_true:
                    continue

                y_true_arr = np.asarray(pooled_true)
                y_pred_arr = np.asarray(pooled_pred)
                summaries.append(
                    {
                        "task": "regression",
                        "track": track_name,
                        "model": model_name,
                        "target": target_name,
                        "horizon": horizon,
                        "n_predictions": int(y_true_arr.size),
                        "n_folds_used": used_folds,
                        "n_features_after_track": feature_count,
                        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
                        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
                    }
                )

    return pd.DataFrame(summaries), pd.DataFrame(predictions)


def evaluate_classification(
    track_name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    folds: list[dict[str, object]],
    include_trees: bool,
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    models = classification_models(include_trees)
    if include_trees is False:
        model_track_allow = {name: set(TRACK_ORDER) for name in models}
    else:
        model_track_allow = {
            "logit": set(TRACK_ORDER),
            "rf": {"B", "D-mini"},
        }

    for model_name, model in models.items():
        if track_name not in model_track_allow[model_name]:
            continue
        for horizon in horizons:
            col = f"y_USREC_h{horizon}"
            if col not in targets.columns:
                continue
            pooled_true: list[int] = []
            pooled_prob: list[float] = []
            used_folds = 0
            feature_count = None
            for fold_id, fold in enumerate(folds):
                X_train, y_train, X_test, y_test = prepare_split(features, targets[col], fold)
                if X_train.empty or X_test.empty:
                    continue
                if y_train.nunique() < 2:
                    continue

                transform = make_track_transform(track_name)
                X_train_fit, X_test_fit = safe_fit_transform(transform, X_train, y_train, X_test)

                if feature_count is None:
                    feature_count = int(X_train_fit.shape[1])

                fitted = clone(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    fitted.fit(X_train_fit, y_train)
                    proba = fitted.predict_proba(X_test_fit)[:, 1]
                if not np.isfinite(proba).all():
                    continue
                pooled_true.extend(y_test.astype(int).tolist())
                pooled_prob.extend(np.asarray(proba).tolist())
                used_folds += 1

                for dt, truth, prob in zip(X_test.index, y_test, proba):
                    predictions.append(
                        {
                            "task": "classification",
                            "track": track_name,
                            "model": model_name,
                            "target": "USREC",
                            "horizon": horizon,
                            "fold": fold_id,
                            "date": str(pd.Timestamp(dt).date()),
                            "y_true": int(truth),
                            "y_prob": float(prob),
                        }
                    )

            if not pooled_true:
                continue

            y_true_arr = np.asarray(pooled_true, dtype=int)
            y_prob_arr = np.clip(np.asarray(pooled_prob), 1e-6, 1 - 1e-6)
            y_hat_arr = (y_prob_arr >= 0.5).astype(int)

            metrics = {
                "task": "classification",
                "track": track_name,
                "model": model_name,
                "target": "USREC",
                "horizon": horizon,
                "n_predictions": int(y_true_arr.size),
                "n_folds_used": used_folds,
                "n_features_after_track": feature_count,
                "positive_rate": float(y_true_arr.mean()),
                "accuracy": float(accuracy_score(y_true_arr, y_hat_arr)),
                "average_precision": float(average_precision_score(y_true_arr, y_prob_arr)),
                "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
                "log_loss": float(log_loss(y_true_arr, y_prob_arr)),
            }
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
            summaries.append(metrics)

    return pd.DataFrame(summaries), pd.DataFrame(predictions)


def render_markdown(
    reg_summary: pd.DataFrame,
    cls_summary: pd.DataFrame,
    args: argparse.Namespace,
    metadata: dict,
) -> str:
    lines = [
        "# Classical Benchmark Starter Run",
        "",
        f"- Generated from `scripts/run_classical_benchmark.py`",
        f"- Tracks: {', '.join(args.tracks)}",
        f"- Horizons: {', '.join(map(str, args.horizons))}",
        f"- Tree baselines included: {'yes' if args.include_trees else 'no'}",
        f"- Folds: {metadata['artifacts']['folds.json']['desc']}",
        "",
    ]

    if not reg_summary.empty:
        reg_filtered = reg_summary[reg_summary["horizon"].isin(args.horizons)].copy()
        best_reg = (
            reg_filtered.sort_values(["target", "horizon", "rmse"])
            .groupby(["target", "horizon"], as_index=False)
            .first()[["target", "horizon", "track", "model", "rmse", "mae"]]
        )
        lines.extend(
            [
                "## Best Regression Rows",
                "",
                best_reg.to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )

    if not cls_summary.empty:
        cls_filtered = cls_summary[cls_summary["horizon"].isin(args.horizons)].copy()
        best_cls = (
            cls_filtered.sort_values(["horizon", "roc_auc"], ascending=[True, False])
            .groupby(["horizon"], as_index=False)
            .first()[["horizon", "track", "model", "roc_auc", "average_precision", "brier"]]
        )
        lines.extend(
            [
                "## Best Classification Rows",
                "",
                best_cls.to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- This is the starter benchmark, not the final capstone leaderboard.",
            "- Track C/D/D-mini are re-fit inside each training fold to avoid leakage.",
            "- USREC metrics pool predictions across folds before scoring, which avoids undefined per-fold ROC-AUC in zero-recession windows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    track_frames, targets, folds, metadata = load_data()

    reg_summaries: list[pd.DataFrame] = []
    reg_predictions: list[pd.DataFrame] = []
    cls_summaries: list[pd.DataFrame] = []
    cls_predictions: list[pd.DataFrame] = []

    for track_name in args.tracks:
        features = track_frames[track_name]
        print(f"[run] track={track_name} rows={features.shape[0]} cols={features.shape[1]}")
        reg_summary, reg_preds = evaluate_regression(
            track_name=track_name,
            features=features,
            targets=targets,
            folds=folds,
            include_trees=args.include_trees,
            horizons=args.horizons,
        )
        cls_summary, cls_preds = evaluate_classification(
            track_name=track_name,
            features=features,
            targets=targets,
            folds=folds,
            include_trees=args.include_trees,
            horizons=args.horizons,
        )

        reg_summaries.append(reg_summary)
        reg_predictions.append(reg_preds)
        cls_summaries.append(cls_summary)
        cls_predictions.append(cls_preds)

    reg_summary = pd.concat([df for df in reg_summaries if not df.empty], ignore_index=True)
    cls_summary = pd.concat([df for df in cls_summaries if not df.empty], ignore_index=True)
    reg_preds = pd.concat([df for df in reg_predictions if not df.empty], ignore_index=True)
    cls_preds = pd.concat([df for df in cls_predictions if not df.empty], ignore_index=True)

    reg_summary = reg_summary[reg_summary["horizon"].isin(args.horizons)].copy()
    cls_summary = cls_summary[cls_summary["horizon"].isin(args.horizons)].copy()
    reg_preds = reg_preds[reg_preds["horizon"].isin(args.horizons)].copy()
    cls_preds = cls_preds[cls_preds["horizon"].isin(args.horizons)].copy()

    reg_summary.sort_values(["target", "horizon", "rmse", "track", "model"]).to_csv(
        args.outdir / "regression_summary.csv",
        index=False,
    )
    cls_summary.sort_values(["target", "horizon", "roc_auc", "track", "model"], ascending=[True, True, False, True, True]).to_csv(
        args.outdir / "classification_summary.csv",
        index=False,
    )
    reg_preds.to_csv(args.outdir / "regression_predictions.csv", index=False)
    cls_preds.to_csv(args.outdir / "classification_predictions.csv", index=False)

    manifest = {
        "tracks": args.tracks,
        "horizons": args.horizons,
        "include_trees": args.include_trees,
        "folds_used": len(folds),
        "seed": SEED,
        "data_vintage": metadata["vintage"],
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.outdir / "SUMMARY.md").write_text(
        render_markdown(reg_summary, cls_summary, args, metadata)
    )

    print("[done] wrote outputs to", args.outdir)


if __name__ == "__main__":
    main()
