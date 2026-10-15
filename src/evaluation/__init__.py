"""Evaluation harness: folds, feature tracks, metrics and the main loop."""
from .folds import generate_folds
from .loop import (
    disaggregate_by_usrec,
    run_evaluation,
    summarize,
    target_col,
)
from .metrics import METRIC_COLS, compute_metrics, naive_mae
from .store import (
    RUN_KEYS,
    completed_runs,
    explode_predictions,
    load_results,
    save_results,
    write_csv,
)
from .tracks import load_tracks, prepare_track, shared_track

__all__ = [
    "generate_folds",
    "run_evaluation",
    "save_results",
    "load_results",
    "completed_runs",
    "explode_predictions",
    "write_csv",
    "RUN_KEYS",
    "summarize",
    "disaggregate_by_usrec",
    "target_col",
    "METRIC_COLS",
    "compute_metrics",
    "naive_mae",
    "load_tracks",
    "prepare_track",
    "shared_track",
]
