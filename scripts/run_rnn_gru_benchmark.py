from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUT_DIR = ROOT / "results" / "william_rnn_gru"
REG_TARGETS = ["INDPRO", "PAYEMS", "CPIAUCSL", "S&P 500"]
ALL_HORIZONS = [1, 3, 6, 12]
TRACKS = ["B", "D-mini"]
SEED = 42


@dataclass
class SequenceData:
    X_train: np.ndarray
    y_train: np.ndarray
    train_dates: list[pd.Timestamp]
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: list[pd.Timestamp]
    feature_names: list[str]


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        model_name: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if model_name == "rnn":
            self.core = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=rnn_dropout,
            )
        elif model_name == "gru":
            self.core = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.core(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run William's fixed-window RNN/GRU macro forecasting benchmark."
    )
    parser.add_argument("--track", choices=TRACKS, default="B")
    parser.add_argument("--targets", nargs="+", default=REG_TARGETS, choices=REG_TARGETS)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1], choices=ALL_HORIZONS)
    parser.add_argument("--models", nargs="+", default=["rnn", "gru"], choices=["rnn", "gru"])
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--train-window-months", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_shared_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]], dict]:
    track_b = pd.read_parquet(DATA_DIR / "track_B_curated.parquet").sort_index()
    stationary = pd.read_parquet(DATA_DIR / "stationary_panel.parquet").sort_index()
    targets = pd.read_parquet(DATA_DIR / "targets.parquet").sort_index()
    folds = json.loads((DATA_DIR / "folds.json").read_text())["shared"]
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    return track_b, stationary, targets, folds, metadata


def truncate_train_range(fold: dict[str, object], train_window_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_end = pd.Timestamp(fold["train_end"])
    train_start = train_end - pd.DateOffset(months=train_window_months - 1)
    return train_start, train_end


def build_monthly_feature_frame(
    track_name: str,
    track_b: pd.DataFrame,
    stationary: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    start_needed = train_start - pd.DateOffset(months=23)
    if track_name == "B":
        frame = track_b.loc[start_needed:test_end].copy()
        frame = frame.drop(columns=["T10Y3M_delta"], errors="ignore")
        return frame, frame.columns.tolist()

    source = stationary.loc[start_needed:test_end].copy()
    fit_source = stationary.loc[train_start:train_end].copy()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=15, random_state=SEED)),
        ]
    )
    pipe.fit(fit_source)
    transformed = pipe.transform(source)
    columns = [f"Dmini_factor_{i+1}" for i in range(transformed.shape[1])]
    frame = pd.DataFrame(transformed, index=source.index, columns=columns)
    return frame, columns


def build_sequence_dataset(
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    sequence_length: int,
    feature_names: list[str],
) -> SequenceData:
    aligned_target = target_series.reindex(feature_frame.index)
    dates = feature_frame.index

    train_X: list[np.ndarray] = []
    train_y: list[float] = []
    train_dates: list[pd.Timestamp] = []
    test_X: list[np.ndarray] = []
    test_y: list[float] = []
    test_dates: list[pd.Timestamp] = []

    for end_pos in range(sequence_length - 1, len(dates)):
        end_ts = dates[end_pos]
        if end_ts < train_start or end_ts > test_end:
            continue
        seq = feature_frame.iloc[end_pos - sequence_length + 1 : end_pos + 1]
        y_val = aligned_target.iloc[end_pos]
        if seq.shape[0] != sequence_length:
            continue
        if not np.isfinite(seq.to_numpy()).all():
            continue
        if pd.isna(y_val) or not np.isfinite(y_val):
            continue
        if train_start <= end_ts <= train_end:
            train_X.append(seq.to_numpy(dtype=np.float32))
            train_y.append(float(y_val))
            train_dates.append(end_ts)
        elif test_start <= end_ts <= test_end:
            test_X.append(seq.to_numpy(dtype=np.float32))
            test_y.append(float(y_val))
            test_dates.append(end_ts)

    return SequenceData(
        X_train=np.asarray(train_X, dtype=np.float32),
        y_train=np.asarray(train_y, dtype=np.float32),
        train_dates=train_dates,
        X_test=np.asarray(test_X, dtype=np.float32),
        y_test=np.asarray(test_y, dtype=np.float32),
        test_dates=test_dates,
        feature_names=feature_names,
    )


def split_train_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_obs = len(X_train)
    if n_obs < 12:
        raise ValueError("Need at least 12 training sequences for RNN/GRU evaluation.")
    val_size = max(4, int(np.ceil(n_obs * 0.2)))
    if val_size >= n_obs:
        val_size = max(1, n_obs // 3)
    split_at = n_obs - val_size
    return X_train[:split_at], y_train[:split_at], X_train[split_at:], y_train[split_at:]


def standardize_sequences(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    feature_mean = X_fit.mean(axis=(0, 1), keepdims=True)
    feature_std = X_fit.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    X_fit_std = (X_fit - feature_mean) / feature_std
    X_val_std = (X_val - feature_mean) / feature_std
    X_test_std = (X_test - feature_mean) / feature_std

    target_mean = float(y_fit.mean())
    target_std = float(y_fit.std())
    if target_std < 1e-6:
        target_std = 1.0

    y_fit_std = (y_fit - target_mean) / target_std
    y_val_std = (y_val - target_mean) / target_std
    return X_fit_std, y_fit_std, X_val_std, y_val_std, X_test_std, target_mean, target_std


def fit_one_model(
    model_name: str,
    sequence_data: SequenceData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    X_fit, y_fit, X_val, y_val = split_train_validation(
        sequence_data.X_train,
        sequence_data.y_train,
    )
    X_fit_std, y_fit_std, X_val_std, y_val_std, X_test_std, y_mean, y_std = standardize_sequences(
        X_fit,
        y_fit,
        X_val,
        y_val,
        sequence_data.X_test,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_fit_std, dtype=torch.float32),
            torch.tensor(y_fit_std, dtype=torch.float32),
        ),
        batch_size=min(args.batch_size, len(X_fit_std)),
        shuffle=True,
    )

    model = RecurrentRegressor(
        input_size=sequence_data.X_train.shape[2],
        hidden_size=args.hidden_size,
        model_name=model_name,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    X_val_tensor = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val_std, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_std, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = float(loss_fn(val_pred, y_val_tensor).cpu().item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(X_test_tensor).cpu().numpy()

    preds = pred_std * y_std + y_mean
    return preds.astype(np.float32), best_val, best_epoch


def render_markdown(
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    metadata: dict,
) -> str:
    lines = [
        "# William Sequence Models",
        "",
        "- Models: " + ", ".join(args.models),
        f"- Track: {args.track}",
        f"- Sequence length: {args.sequence_length}",
        f"- Fixed train window: {args.train_window_months} months",
        f"- Tested horizons: {', '.join(map(str, args.horizons))}",
        f"- Folds available in substrate: {metadata['artifacts']['folds.json']['desc']}",
        "",
    ]
    if not summary_df.empty:
        best_rows = (
            summary_df.sort_values(["target", "horizon", "rmse"])
            .groupby(["target", "horizon"], as_index=False)
            .first()[["target", "horizon", "model", "rmse", "mae", "n_predictions", "mean_best_epoch"]]
        )
        header = "| target | horizon | model | rmse | mae | n_predictions | mean_best_epoch |"
        divider = "| --- | ---: | --- | ---: | ---: | ---: | ---: |"
        body = [
            (
                f"| {row.target} | {int(row.horizon)} | {row.model} | "
                f"{row.rmse:.6f} | {row.mae:.6f} | {int(row.n_predictions)} | "
                f"{row.mean_best_epoch:.6f} |"
            )
            for row in best_rows.itertuples(index=False)
        ]
        lines.extend(
            [
                "## Best Rows",
                "",
                header,
                divider,
                *body,
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.",
            "- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.",
            "- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = choose_device()
    track_b, stationary, targets, folds, metadata = load_shared_data()
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for target_name in args.targets:
        for horizon in args.horizons:
            target_col = f"y_{target_name}_h{horizon}"
            target_series = targets[target_col]
            for model_name in args.models:
                pooled_truth: list[float] = []
                pooled_pred: list[float] = []
                best_epochs: list[int] = []
                val_losses: list[float] = []
                n_train_samples: list[int] = []
                n_test_samples: list[int] = []

                for fold_id, fold in enumerate(folds):
                    train_start, train_end = truncate_train_range(fold, args.train_window_months)
                    test_start = pd.Timestamp(fold["test_start"])
                    test_end = pd.Timestamp(fold["test_end"])

                    feature_frame, feature_names = build_monthly_feature_frame(
                        track_name=args.track,
                        track_b=track_b,
                        stationary=stationary,
                        train_start=train_start,
                        train_end=train_end,
                        test_end=test_end,
                    )
                    seq_data = build_sequence_dataset(
                        feature_frame=feature_frame,
                        target_series=target_series,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        sequence_length=args.sequence_length,
                        feature_names=feature_names,
                    )

                    if len(seq_data.X_train) < 12 or len(seq_data.X_test) == 0:
                        continue

                    preds, val_loss, best_epoch = fit_one_model(
                        model_name=model_name,
                        sequence_data=seq_data,
                        args=args,
                        device=device,
                    )

                    pooled_truth.extend(seq_data.y_test.tolist())
                    pooled_pred.extend(preds.tolist())
                    best_epochs.append(best_epoch)
                    val_losses.append(val_loss)
                    n_train_samples.append(len(seq_data.X_train))
                    n_test_samples.append(len(seq_data.X_test))

                    for dt, truth, pred in zip(seq_data.test_dates, seq_data.y_test, preds):
                        prediction_rows.append(
                            {
                                "track": args.track,
                                "model": model_name,
                                "target": target_name,
                                "horizon": horizon,
                                "fold": fold_id,
                                "date": str(pd.Timestamp(dt).date()),
                                "y_true": float(truth),
                                "y_pred": float(pred),
                            }
                        )

                if not pooled_truth:
                    continue

                y_true = np.asarray(pooled_truth)
                y_pred = np.asarray(pooled_pred)
                summary_rows.append(
                    {
                        "track": args.track,
                        "model": model_name,
                        "target": target_name,
                        "horizon": horizon,
                        "n_predictions": int(y_true.size),
                        "n_folds_used": len(best_epochs),
                        "avg_train_sequences": float(np.mean(n_train_samples)),
                        "avg_test_sequences": float(np.mean(n_test_samples)),
                        "mean_best_epoch": float(np.mean(best_epochs)),
                        "mean_val_loss": float(np.mean(val_losses)),
                        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        "mae": float(mean_absolute_error(y_true, y_pred)),
                    }
                )
                print(
                    f"[done] target={target_name} h={horizon} model={model_name} "
                    f"rmse={summary_rows[-1]['rmse']:.6f} mae={summary_rows[-1]['mae']:.6f}"
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["target", "horizon", "rmse", "model"]
    )
    pred_df = pd.DataFrame(prediction_rows).sort_values(
        ["target", "horizon", "model", "fold", "date"]
    )

    summary_df.to_csv(args.outdir / "summary.csv", index=False)
    pred_df.to_csv(args.outdir / "predictions.csv", index=False)
    manifest = {
        "track": args.track,
        "targets": args.targets,
        "horizons": args.horizons,
        "models": args.models,
        "sequence_length": args.sequence_length,
        "train_window_months": args.train_window_months,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_folds": args.max_folds,
        "seed": SEED,
        "device": str(device),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.outdir / "SUMMARY.md").write_text(render_markdown(summary_df, args, metadata))


if __name__ == "__main__":
    main()
