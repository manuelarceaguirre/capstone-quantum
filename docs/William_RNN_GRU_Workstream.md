# William Workstream Update

## Scope

This note summarizes the recurrent-model workstream for the BMO capstone.
William's lane focuses on sequence models that can serve as the classical temporal
benchmark closest to the reservoir-computing intuition:

- vanilla `RNN`
- gated `GRU`

The goal is not to replace the quantum reservoir story. The goal is to create a
clean classical recurrent reference point before the team claims any quantum gain.

## Constraints Taken From The Repository

- The official shared substrate is `notebooks/01_data_prep.ipynb`.
- All downstream models must load artifacts from `data/processed/`.
- Leakage control matters: anything like PCA must be fit inside each fold only.
- The repository ships with `39` shared expanding-window test folds.
- For the recurrent lane, we intentionally replace the expanding training history
  with a fixed-length training window because memory-based models are sensitive to
  window size and training cost.

## What Was Added

- `scripts/run_rnn_gru_benchmark.py`

This script:

- loads Track B or fold-wise D-mini features from the shared processed artifacts
- truncates each training fold to a constant history length
- builds rolling sequences of fixed length
- trains `RNN` and `GRU` models with temporal validation splits
- exports pooled out-of-sample predictions and summary metrics

## Modeling Choice

The default setup is:

- `Track B` first, because it is low-dimensional and easy to explain
- `sequence_length = 24` months
- `train_window_months = 120`
- pooled evaluation across shared test folds

This makes the recurrent lane easy to compare against:

- Jack's statistical baselines
- David's signature-kernel lane
- Manuel's Konduri-style benchmark recreation
- future QRC experiments

## Why Fixed Windows

The repository's default evaluation substrate uses expanding folds. That is still
the right choice for cross-team alignment on test dates.

For RNN/GRU and reservoir-style models, however, using the entire historical
training path creates two practical problems:

- training cost grows fold by fold
- model memory depends too heavily on early-regime data that may no longer be
  relevant

So the implementation keeps the same test periods but truncates the training side
to a constant rolling history. This preserves fair date alignment while making the
sequence models operationally realistic.

## Recommended Experiment Order

1. Start with `Track B`, horizon `h=1`, all four regression targets.
2. Compare `RNN` vs `GRU` using the same fixed window.
3. Move to `Track D-mini` once the Track B run is stable.
4. Only then decide whether a classical reservoir or QRC comparison is justified.

## Full Track B Results

Using the `CV_Work` environment, the formal Track B run was completed on:

- `Track B`
- horizons: `h=1`
- all `39` shared folds
- fixed train window: `120` months
- sequence length: `24`
- max epochs: `30`
- early-stopping patience: `6`

Full pooled out-of-sample results:

| Target | Best model | RMSE | MAE |
| --- | --- | ---: | ---: |
| `INDPRO` | `RNN` | `0.010032` | `0.005375` |
| `PAYEMS` | `RNN` | `0.007219` | `0.001763` |
| `CPIAUCSL` | `GRU` | `0.003004` | `0.002183` |
| `S&P 500` | `RNN` | `0.037534` | `0.026956` |

Each row pools `2,340` predictions across the shared test windows. The result is
useful as William's official first-pass recurrent benchmark, not as a final model
selection claim. The most important signal is that `GRU` does not dominate the
simpler `RNN`; it only wins on `CPIAUCSL` in this configuration.

## Additional Quick Checks

Before the full run, a quick first-5-fold Track B check was used to validate the
pipeline end to end. An initial `D-mini` comparison for `INDPRO, h=1` was also run:

- `GRU`: RMSE `0.005398`, MAE `0.004137`
- `RNN`: RMSE `0.005656`, MAE `0.004477`

These quick checks are archived as exploratory diagnostics. The full Track B run
above should be used for group discussion.

## Interpretation For Group Discussion

- `Track B` is now a viable first recurrent lane because it is low-dimensional,
  interpretable, and fully aligned with the team's shared fold structure.
- `RNN` is currently competitive with or better than `GRU` on three of the four
  targets, so the team should not assume gating automatically improves limited-data
  macro forecasting.
- `D-mini` remains valuable for QRC alignment, but it should be treated as the next
  comparison lane rather than the only recurrent input.

## Run Command

Use the `CV_Work` environment:

```bash
~/.virtualenvs/CV_Work/bin/python scripts/run_rnn_gru_benchmark.py \
  --track B \
  --targets INDPRO PAYEMS CPIAUCSL "S&P 500" \
  --horizons 1 \
  --epochs 30 \
  --outdir results/william_rnn_gru
```

Quick sanity run:

```bash
~/.virtualenvs/CV_Work/bin/python scripts/run_rnn_gru_benchmark.py \
  --track B \
  --targets INDPRO \
  --horizons 1 \
  --epochs 12 \
  --max-folds 5 \
  --outdir results/william_rnn_gru_sanity
```

Artifacts already written:

- `results/william_rnn_gru/summary.csv`
- `results/william_rnn_gru/SUMMARY.md`
- `results/william_rnn_gru/predictions.csv`
- `results/william_rnn_gru_quick/summary.csv`
- `results/william_rnn_gru_quick/SUMMARY.md`
- `results/william_rnn_gru_dmini_quick/summary.csv`

## Discussion Points For The Team

- Should William's lane stay purely predictive with `RNN/GRU`, or should it also
  include a classical reservoir implementation later?
- Should Track B remain the main recurrent input, or should D-mini become the main
  comparison lane for QRC alignment?
- Do we want one shared metric table for all workstreams, or separate benchmark
  tables for regression and classification?
