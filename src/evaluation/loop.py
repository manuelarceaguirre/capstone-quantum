"""The evaluation loop: tracks x folds x targets x horizons x models."""
import getpass
import time

import numpy as np
import pandas as pd

from .metrics import METRIC_COLS, compute_metrics, naive_mae
from .store import completed_runs, save_results
from .tracks import prepare_track

GROUP_COLS = ["model", "track", "target", "horizon"]


def target_col(target, horizon):
    """Column name of the h-step target in targets.parquet."""
    return f"y_{target}_h{horizon}"


def _fmt_duration(seconds):
    """Format seconds as e.g. '45s', '4m12s', '2h05m'."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


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
    run_by=None,
    results_dir=None,
    checkpoint_every=25,
    resume=False,
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
    run_by           : name recorded in the ``run_by`` column (default: OS user)
    results_dir      : if given, rows are merged into results_dir via save_results()
                       every ``checkpoint_every`` folds and at the end of each track
    checkpoint_every : folds between checkpoints when results_dir is given
    resume           : skip runs that already have a successful result with predictions
                       in results_dir; errored and prediction-less (legacy) runs are redone.
                       Leave off after changing a model, or its old results are kept.

    Returns
    -------
    pd.DataFrame with one row per run computed in this call (skipped runs are not
    included): identifiers, metrics, runtime, error, and ``y_true`` / ``y_pred``
    lists over the test window (see explode_predictions)
    """
    if resume and results_dir is None:
        raise ValueError("resume=True needs results_dir")

    run_by = run_by or getpass.getuser()
    run_at = pd.Timestamp.now(tz="UTC")
    results = []
    total_runs = len(folds) * len(tracks) * len(targets) * len(horizons) * len(models)
    completed = 0
    skipped = 0  # runs already on disk (resume), excluded from the ETA
    fold_ids = {fold["fold_id"] for fold in folds}
    t_start = time.time()

    for track_name in tracks:
        done = completed_runs(results_dir, models, track_name) if resume else set()
        if done:
            print(f"  Track {track_name}: resuming, {len(done)} runs already on disk will be skipped")
            skipped += sum(
                1 for model_name, target, horizon, fold_id in done
                if model_name in models and target in targets
                and horizon in horizons and fold_id in fold_ids
            )
        pending = []  # rows computed since the last checkpoint

        for i, fold in enumerate(folds, 1):
            for target in targets:
                todo = {
                    (horizon, model_name)
                    for horizon in horizons
                    for model_name in models
                    if (model_name, target, horizon, fold["fold_id"]) not in done
                }
                if not todo:
                    continue

                # slice and prepare track data once per track/fold/target
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
                        if (horizon, model_name) not in todo:
                            continue
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
                            "y_true":      y_test.astype(float).tolist(),
                            "y_pred":      None,
                            "run_at":      run_at,
                            "run_by":      run_by,
                        }

                        t0 = time.time()
                        try:
                            y_pred = model_fn(train_df, test_df, y_train, horizon)
                            row["runtime_sec"] = round(time.time() - t0, 4)
                            row["y_pred"] = np.asarray(y_pred, dtype=float).tolist()
                            row.update(compute_metrics(y_test.values, y_pred.values, scale))
                        except Exception as e:
                            row["runtime_sec"] = round(time.time() - t0, 4)
                            row.update({m: np.nan for m in METRIC_COLS})
                            row["error"] = str(e)

                        pending.append(row)
                        completed += 1
                        if progress_every and completed % progress_every == 0:
                            elapsed = time.time() - t_start
                            remaining = max(total_runs - skipped - completed, 0)
                            eta = elapsed / completed * remaining
                            print(
                                f"  {completed + skipped}/{total_runs} runs complete"
                                f" · {_fmt_duration(elapsed)} elapsed"
                                f" · ~{_fmt_duration(eta)} left"
                            )

            # checkpoint every few folds and at the end of the track, so a crash
            # loses at most checkpoint_every folds of work
            if results_dir is not None and pending and (i % checkpoint_every == 0 or i == len(folds)):
                save_results(pd.DataFrame(pending), results_dir)
                results.extend(pending)
                pending = []
        results.extend(pending)

    print(f"Loop complete. Total rows: {len(results)} in {_fmt_duration(time.time() - t_start)}")
    return pd.DataFrame(results)


def summarize(results_df):
    """Mean metrics across folds per model/track/target/horizon."""
    return results_df.groupby(GROUP_COLS)[METRIC_COLS].mean().round(6)


def disaggregate_by_usrec(results_df, usrec):
    """Mean metrics split by recession (1) vs expansion (0) at the test date."""
    usrec_clean = usrec.rename(columns={usrec.columns[0]: "usrec"})
    merged = results_df.merge(usrec_clean, left_on="test_start", right_index=True, how="left")
    return merged.groupby(GROUP_COLS + ["usrec"])[METRIC_COLS].mean().round(6)
