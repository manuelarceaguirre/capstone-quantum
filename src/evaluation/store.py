"""Result storage that is safe for several people running models at once.

Layout
------
``results/raw/<model>/<track>.parquet`` — one file per (model, track), so people
running different models or tracks never write the same file (locally or in git).

Writes are merges, not overwrites: rows are keyed by ``RUN_KEYS`` and a new run
replaces only the rows it recomputed. A 5-fold smoke test therefore updates those
5 folds and leaves a full run's other folds in place. Each write holds a file lock
and replaces the file atomically, so two processes on the same machine saving the
same (model, track) cannot corrupt or drop each other's rows.
"""
import os
from pathlib import Path

import pandas as pd
from filelock import FileLock

RUN_KEYS = ["model", "track", "target", "horizon", "fold_id"]


def _partition_path(results_dir, model, track):
    return Path(results_dir) / "raw" / model / f"{track}.parquet"


def _atomic_write(df, path, writer):
    """Write to a temp file in the same directory, then rename over the target."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    writer(df, tmp)
    os.replace(tmp, path)


def _merge_into(path, new_df):
    """Upsert new_df into the parquet at path under a lock. Returns total rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            old_df = pd.read_parquet(path)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(RUN_KEYS, keep="last")
        else:
            merged = new_df
        merged = merged.sort_values(RUN_KEYS).reset_index(drop=True)
        _atomic_write(merged, path, lambda df, p: df.to_parquet(p, index=False))
    return len(merged)


def save_results(results_df, results_dir):
    """Merge results into results/raw/<model>/<track>.parquet."""
    for (model, track), part in results_df.groupby(["model", "track"], sort=False):
        path = _partition_path(results_dir, model, track)
        total = _merge_into(path, part)
        print(f"  Saved: raw/{model}/{track}.parquet   new={len(part)}  total={total}")


def load_results(results_dir, models=None, tracks=None):
    """
    Read stored results for every (model, track) on disk — including runs made by
    other people — optionally filtered to the given models / tracks.
    """
    paths = sorted((Path(results_dir) / "raw").glob("*/*.parquet"))
    paths = [
        p for p in paths
        if (models is None or p.parent.name in models)
        and (tracks is None or p.stem in tracks)
    ]
    if not paths:
        raise FileNotFoundError(f"No results under {Path(results_dir) / 'raw'}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def completed_runs(results_dir, models, track):
    """
    Keys ``(model, target, horizon, fold_id)`` of runs on disk for this track that
    succeeded and have stored predictions — the runs a resumed loop can skip.
    """
    keys = set()
    for model in models:
        path = _partition_path(results_dir, model, track)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["model", "target", "horizon", "fold_id", "error", "y_pred"])
        ok = df[df["error"].isna() & df["y_pred"].notna()]
        keys.update(
            (m, t, int(h), int(f))
            for m, t, h, f in zip(ok["model"], ok["target"], ok["horizon"], ok["fold_id"])
        )
    return keys


def write_csv(df, path):
    """Atomically write a derived table (e.g. a summary) so readers never see a partial file."""
    path = Path(path)
    with FileLock(str(path) + ".lock"):
        _atomic_write(df, path, lambda d, p: d.to_csv(p))


def explode_predictions(results_df):
    """
    One row per (run, test date) with columns
    ``model, track, target, horizon, fold_id, date, y_true, y_pred``.
    Runs without stored predictions (errors, legacy rows) are dropped.
    """
    df = results_df.dropna(subset=["y_pred"])
    rows = []
    for r in df.itertuples(index=False):
        dates = pd.date_range(r.test_start, r.test_end, freq="MS")
        for date, yt, yp in zip(dates, r.y_true, r.y_pred):
            rows.append((r.model, r.track, r.target, r.horizon, r.fold_id, date, yt, yp))
    return pd.DataFrame(rows, columns=RUN_KEYS + ["date", "y_true", "y_pred"])
