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
- `train_window_months = 24`
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
- horizons: `h=1, 3, 6, 12`
- all `39` shared folds
- fixed train window: `24` months
- sequence length: `24`
- max epochs: `30`
- early-stopping patience: `6`
- metrics: `RMSE`, `MAE`, and `MASE`

Full pooled out-of-sample results:

| Target | Horizon | Model | RMSE | MAE | MASE | Predictions |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `CPIAUCSL` | 1 | `GRU` | `0.003002` | `0.002130` | `1.205221` | 2340 |
| `CPIAUCSL` | 1 | `RNN` | `0.002872` | `0.001990` | `1.152910` | 2340 |
| `CPIAUCSL` | 3 | `GRU` | `0.007130` | `0.004825` | `2.185147` | 2340 |
| `CPIAUCSL` | 3 | `RNN` | `0.007293` | `0.005131` | `2.298783` | 2340 |
| `CPIAUCSL` | 6 | `GRU` | `0.011428` | `0.008095` | `3.551565` | 2340 |
| `CPIAUCSL` | 6 | `RNN` | `0.011771` | `0.008167` | `3.571194` | 2340 |
| `CPIAUCSL` | 12 | `GRU` | `0.018646` | `0.013722` | `5.821866` | 2339 |
| `CPIAUCSL` | 12 | `RNN` | `0.018621` | `0.013652` | `5.754534` | 2339 |
| `INDPRO` | 1 | `GRU` | `0.010523` | `0.005849` | `1.081108` | 2340 |
| `INDPRO` | 1 | `RNN` | `0.010444` | `0.005748` | `1.067959` | 2340 |
| `INDPRO` | 3 | `GRU` | `0.021525` | `0.013407` | `2.453268` | 2340 |
| `INDPRO` | 3 | `RNN` | `0.021054` | `0.012888` | `2.407835` | 2340 |
| `INDPRO` | 6 | `GRU` | `0.033079` | `0.023051` | `3.913385` | 2340 |
| `INDPRO` | 6 | `RNN` | `0.033177` | `0.023526` | `4.016241` | 2340 |
| `INDPRO` | 12 | `GRU` | `0.054001` | `0.039088` | `5.771831` | 2339 |
| `INDPRO` | 12 | `RNN` | `0.056336` | `0.040747` | `5.975287` | 2339 |
| `PAYEMS` | 1 | `GRU` | `0.007384` | `0.002076` | `2.415289` | 2340 |
| `PAYEMS` | 1 | `RNN` | `0.007368` | `0.002024` | `2.388242` | 2340 |
| `PAYEMS` | 3 | `GRU` | `0.012867` | `0.005521` | `6.332766` | 2340 |
| `PAYEMS` | 3 | `RNN` | `0.012845` | `0.005502` | `6.320277` | 2340 |
| `PAYEMS` | 6 | `GRU` | `0.018458` | `0.010718` | `9.803875` | 2340 |
| `PAYEMS` | 6 | `RNN` | `0.019183` | `0.010979` | `9.833723` | 2340 |
| `PAYEMS` | 12 | `GRU` | `0.029853` | `0.020596` | `15.198590` | 2339 |
| `PAYEMS` | 12 | `RNN` | `0.029915` | `0.020408` | `15.142995` | 2339 |
| `S&P 500` | 1 | `GRU` | `0.039564` | `0.028407` | `0.963306` | 2340 |
| `S&P 500` | 1 | `RNN` | `0.038689` | `0.027981` | `0.948989` | 2340 |
| `S&P 500` | 3 | `GRU` | `0.081012` | `0.059438` | `1.826745` | 2340 |
| `S&P 500` | 3 | `RNN` | `0.081740` | `0.060650` | `1.837380` | 2340 |
| `S&P 500` | 6 | `GRU` | `0.132389` | `0.100089` | `2.993925` | 2340 |
| `S&P 500` | 6 | `RNN` | `0.133941` | `0.099248` | `2.943822` | 2340 |
| `S&P 500` | 12 | `GRU` | `0.200795` | `0.151278` | `4.249884` | 2339 |
| `S&P 500` | 12 | `RNN` | `0.199421` | `0.152069` | `4.352434` | 2339 |

Each row pools predictions across the shared test windows. Horizons `1, 3, and 6`
have `2,340` predictions; horizon `12` has `2,339` because the last target label
falls beyond the available data range. MASE is computed fold by fold using the
training target's naive one-step MAE as the denominator, then pooled over test
predictions. The result is useful as William's official recurrent benchmark, not
as a final model-selection claim.

## Additional Quick Checks

Before the full run, a quick first-5-fold Track B check was used to validate the
pipeline end to end. An updated `D-mini` comparison for `INDPRO, h=1` was also run
with the same 24-month training window:

- `GRU`: RMSE `0.006905`, MAE `0.005443`
- `RNN`: RMSE `0.007616`, MAE `0.005849`

For the same D-mini check, MASE is `0.912718` for `GRU` and `0.939995` for `RNN`.

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
