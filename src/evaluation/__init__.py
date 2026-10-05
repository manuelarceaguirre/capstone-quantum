"""Evaluation harness: folds, feature tracks, metrics and the main loop."""
from .folds import generate_folds
from .loop import (
    disaggregate_by_usrec,
    run_evaluation,
    save_results,
    summarize,
    target_col,
)
from .metrics import METRIC_COLS, compute_metrics, naive_mae
from .tracks import load_tracks, prepare_track, shared_track

__all__ = [
    "generate_folds",
    "run_evaluation",
    "save_results",
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
