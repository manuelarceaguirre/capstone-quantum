"""Precompute reservoir feature tracks for the evaluation loop.

For each target, the Track B predictors are fed through
  - QRC: a quantum reservoir (src.features.quantum.quantum_reservoir), and
  - CRC: a classical untrained-RNN reservoir of the same output width (control),
and the final reservoir state for the trailing ``--window`` months ending at
each date becomes that date's feature row:

    data/processed/track_QRC_<target>.parquet
    data/processed/track_CRC_<target>.parquet

Why precompute instead of fitting per fold: the reservoirs are untrained and
each row depends only on its own trailing window, so a row's features never
change between folds. The one leakage risk is input scaling, so inputs are
min-max scaled with expanding (causal) statistics, never the full sample.

Usage:
    python scripts/build_quantum_tracks.py                         # all targets, both kinds
    python scripts/build_quantum_tracks.py --target INDPRO --kind qrc
    python scripts/build_quantum_tracks.py --end 1975-12-01 --out-dir <dir>   # quick smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TARGETS = ["INDPRO", "PAYEMS", "CPIAUCSL", "S&P 500"]


def causal_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each column to [0, 1] using min/max over rows up to and including t."""
    lo = df.expanding().min()
    hi = df.expanding().max()
    span = (hi - lo).replace(0, np.nan)
    return ((df - lo) / span).fillna(0.5).clip(0, 1)


def trailing_windows(n_rows: int, window: int) -> list[list[int]]:
    """Row-index sequences [t-window+1, ..., t] for every t with a full window."""
    return [list(range(t - window + 1, t + 1)) for t in range(window - 1, n_rows)]


def build_qrc(scaled: pd.DataFrame, series, hidden_qubits: int, num_features: int, seed: int) -> pd.DataFrame:
    from qiskit.circuit.library import efficient_su2

    from src.features.quantum.encoding_strategy.su_encoding_strategy import su_encoding_strategy
    from src.features.quantum.quantum_reservoir import quantum_reservoir
    from src.features.quantum.result_getters.pce_result_getter import pce_result_getter

    su2_gates = ["ry", "rx"]
    encoding_qubits = int(np.ceil(scaled.shape[1] / len(su2_gates)))
    n_qubits = encoding_qubits + hidden_qubits

    ansatz = efficient_su2(n_qubits, reps=1)
    rng = np.random.default_rng(seed)
    ansatz.assign_parameters((2 * np.pi * rng.random(ansatz.num_parameters)).tolist(), inplace=True)

    encoding_strategy = su_encoding_strategy(
        n_qubits=encoding_qubits, reps=1, su2_gates=su2_gates, entanglement="full", scale=False
    )
    result_getter = pce_result_getter(num_qubits=hidden_qubits, num_features=num_features, num_operators=2)

    features = quantum_reservoir(
        data=scaled,
        series_indicies=series,
        encoding_strategy=encoding_strategy,
        ansatz=ansatz,
        result_getter=result_getter,
    )
    features.columns = [f"qrc_{c}" for c in features.columns]
    return features


def build_crc(scaled: pd.DataFrame, series, num_features: int, seed: int) -> pd.DataFrame:
    import torch

    from src.features.quantum.classical_reservoir import classical_reservoir

    torch.manual_seed(seed)
    features = classical_reservoir(data=scaled, series_indicies=series, num_hidden_features=num_features)
    features.columns = [f"crc_{c}" for c in features.columns]
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=TARGETS, action="append", help="repeatable; default all")
    parser.add_argument("--kind", choices=["qrc", "crc", "both"], default="both")
    parser.add_argument("--window", type=int, default=12, help="reservoir sequence length in months")
    parser.add_argument("--hidden-qubits", type=int, default=3)
    parser.add_argument("--num-features", type=int, default=9, help="output columns per track")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end", help="truncate input at this date (smoke tests)")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    kinds = ["qrc", "crc"] if args.kind == "both" else [args.kind]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for target in args.target or TARGETS:
        inputs = pd.read_parquet(DATA_DIR / f"track_b_{target}.parquet").sort_index()
        if args.end:
            inputs = inputs.loc[: args.end]
        scaled = causal_minmax(inputs)
        series = trailing_windows(len(scaled), args.window)
        index = scaled.index[args.window - 1:]

        for kind in kinds:
            t0 = time.time()
            if kind == "qrc":
                features = build_qrc(scaled, series, args.hidden_qubits, args.num_features, args.seed)
            else:
                features = build_crc(scaled, series, args.num_features, args.seed)
            features.index = index

            path = args.out_dir / f"track_{kind.upper()}_{target}.parquet"
            features.to_parquet(path)
            print(f"wrote {path.name} shape={features.shape} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
