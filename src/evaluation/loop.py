"""The evaluation loop: folds x tracks x targets x horizons x models."""
import time

import numpy as np
import pandas as pd

from .metrics import METRIC_COLS, compute_metrics, naive_mae
from .tracks import prepare_track

GROUP_COLS = ["model", "track", "target", "horizon"]


def target_col(target, horizon):
    """Column name of the h-step target in targets.parquet."""
    return f"y_{target}_h{horizon}"


def run_evaluation(
    folds,
    tracks,
    targets,
    horizons,
    models,
    targets_df,
    track_config,
    track_data,
    stationary_panel,
    progress_every=1000,
):
    """
    Run every model on every (fold, track, target, horizon) combination.

    Parameters
    ----------
    folds            : list of fold dicts from generate_folds()
    tracks           : list of track names (keys of track_config)
    targets          : list of target names, e.g. ["INDPRO", "S&P 500"]
    horizons         : list of int
    models           : dict name -> model function (see src.models)
    targets_df       : DataFrame of y_<target>_h<h> columns
    track_config     : see src.evaluation.tracks
    track_data       : output of load_tracks()
    stationary_panel : DataFrame used as input to pipeline tracks

    Returns
    -------
    pd.DataFrame with one row per run: identifiers, metrics, runtime, error
    """
    results = []
    total_runs = len(folds) * len(tracks) * len(targets) * len(horizons) * len(models)
    completed = 0

    for fold in folds:
        for track_name in tracks:
            for target in targets:
                # slice and prepare track data once per fold/track/target
                train_df, test_df = prepare_track(
                    track_name, target, fold, track_config, track_data, stationary_panel
                )

                for horizon in horizons:
                    col = target_col(target, horizon)
                    y_train = targets_df.loc[train_df.index, col]
                    y_test  = targets_df.loc[test_df.index, col]

                    # NaN guard — skip if target unavailable
                    if y_test.isna().any() or y_train.isna().any():
                        continue

                    scale = naive_mae(y_train, horizon)

                    for model_name, model_fn in models.items():
                        row = {
                            "model":       model_name,
                            "track":       track_name,
                            "target":      target,
                            "horizon":     horizon,
                            "fold_id":     fold["fold_id"],
                            "train_start": fold["train_start"],
                            "train_end":   fold["train_end"],
                            "test_start":  fold["test_start"],
                            "test_end":    fold["test_end"],
                            "error":       None,
                        }

                        try:
                            t0 = time.time()
                            y_pred = model_fn(train_df, test_df, y_train, horizon)
                            row["runtime_sec"] = round(time.time() - t0, 4)
                            row.update(compute_metrics(y_test.values, y_pred.values, scale))
                        except Exception as e:
                            row["runtime_sec"] = np.nan
                            row.update({m: np.nan for m in METRIC_COLS})
                            row["error"] = str(e)

                        results.append(row)
                        completed += 1
                        if progress_every and completed % progress_every == 0:
                            print(f"  {completed}/{total_runs} runs complete")

    print(f"Loop complete. Total rows: {len(results)}")
    return pd.DataFrame(results)


def save_results(results_df, results_dir):
    """Write one <model>_raw.parquet per model."""
    for model_name in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model_name]
        out_path = results_dir / f"{model_name}_raw.parquet"
        model_df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path.name}   rows={len(model_df)}")


def summarize(results_df):
    """Mean metrics across folds per model/track/target/horizon."""
    return results_df.groupby(GROUP_COLS)[METRIC_COLS].mean().round(6)


def disaggregate_by_usrec(results_df, usrec):
    """Mean metrics split by recession (1) vs expansion (0) at the test date."""
    usrec_clean = usrec.rename(columns={usrec.columns[0]: "usrec"})
    merged = results_df.merge(usrec_clean, left_on="test_start", right_index=True, how="left")
    return merged.groupby(GROUP_COLS + ["usrec"])[METRIC_COLS].mean().round(6)
