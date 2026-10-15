# Capstone — Predicting Macroeconomic Indicators with Classical and Quantum-Enhanced Models

UChicago MS-ADS capstone · Spring 2026 · Sponsor: Bank of Montreal (BMO)

BMO wants to know when (and whether) quantum-enhanced machine learning beats classical
state-of-the-art at forecasting U.S. macro indicators. We replicate and extend
Konduri & Li (2024) on FRED-MD, and benchmark every model on identical data and
identical pseudo-out-of-sample folds so the scores are directly comparable.

- Targets: `INDPRO`, `PAYEMS`, `CPIAUCSL`, `S&P 500` (h-step log growth), horizons 1, 3, 6
- Quantum methods in scope: Quantum Reservoir Computing (QRC) and Quantum Enhanced
  Signature Kernels (QESK)

## How the repo works

One harness, three pluggable pieces:

```
FRED-MD ──01_data_prep──▶ data/processed/ ──┐
                                            ├─▶ 02_evaluation_loop ──▶ results/
         scripts/build_*  ─▶ feature tracks ┘        ▲
                                       src/models/ ──┘ (one function per model)
```

1. **Feature tracks** are the inputs a model sees. Classical tracks (full panel, curated,
   PCA) and quantum tracks (reservoir embeddings) are interchangeable.
2. **Models** are plain functions with a single signature. They never see folds or test targets.
3. **The evaluation loop** iterates folds × tracks × targets × horizons × models, computes
   metrics, and merges the runs into one parquet per (model, track).

## Layout

```
capstone-quantum/
├── data/
│   ├── raw/2026-03-MD.csv            FRED-MD, March 2026 vintage
│   └── processed/                    generated; committed so the loop runs without re-prep
├── notebooks/
│   ├── 01_data_prep.ipynb            cleaning, stationarity, tracks A–D, targets
│   ├── 02_evaluation_loop.ipynb      configuration + run (logic lives in src/evaluation)
│   └── quantum_workflow_example.ipynb  tutorial for the quantum feature tools
├── scripts/
│   ├── build_track_b.py              per-target curated predictor sets (Track B)
│   └── build_quantum_tracks.py       QRC + classical-reservoir (CRC) tracks
├── src/
│   ├── evaluation/                   folds, track prep, metrics, loop
│   ├── models/                       model functions + MODEL_REGISTRY
│   └── features/quantum/             encoding strategies, result getters, reservoirs
└── results/                          raw/<model>/<track>.parquet, summary_disaggregated.csv
```

## Setup

```bash
conda env create -f environment.yml   # includes `pip install -e .`, so `import src...` works anywhere
conda activate capstone
```

In Colab, set `COLAB = True` in the first cell of `02_evaluation_loop.ipynb`.

## Running

```bash
# 1. (only if FRED-MD or prep logic changes) regenerate data/processed
#    run notebooks/01_data_prep.ipynb top to bottom, then:
python scripts/build_track_b.py

# 2. (optional) build quantum / classical reservoir tracks — slow, ~20 min per target for QRC
python scripts/build_quantum_tracks.py

# 3. edit ACTIVE_TRACKS / ACTIVE_MODELS in notebooks/02_evaluation_loop.ipynb and run it
```

Set `MAX_FOLDS = 5` in the config cell for a quick smoke test.

## Results and running in parallel

Each result row is one run (model × track × target × horizon × fold) and holds the
metrics, `runtime_sec`, the test-window `y_true` / `y_pred` (lists), `error`, and
`run_at` / `run_by` (set `RUN_BY` in the notebook config). `explode_predictions()` turns
the lists into one row per test date.

Results are stored as `results/raw/<model>/<track>.parquet`, and saving merges rather
than overwrites: a run replaces only the rows it recomputed (keyed on model, track,
target, horizon, fold_id). So team members can run different models or tracks at the
same time, locally or in Colab, and commit without touching each other's files, and a
smoke test on a few folds doesn't wipe out a full run. Writes hold a file lock and are
atomic, so concurrent saves on one machine are safe too. The loop checkpoints every
25 folds and at the end of each track, so a crash or Colab disconnect loses at most 25
folds of work. To pick up where it stopped, set `RESUME = True` and rerun: runs that already
succeeded with saved predictions are skipped, while errored runs and older results without
predictions are redone. Leave `RESUME = False` after changing a model's code, otherwise
its old results are kept. Two people committing the same
(model, track) file will still conflict in git; agree on who owns each pair.

The summary cells read everything on disk via `load_results()`, not just the latest run.
`summary_disaggregated.csv` is derived, so regenerate it rather than merging it by hand.

## Adding a model

```python
def my_model_function(train_df, test_df, y_train, horizon) -> pd.Series:
    """Return predictions indexed by test_df.index."""
```

- `train_df`, `test_df`: feature rows of one track for this fold (already fit/transformed on train only)
- `y_train`: the horizon-specific target over the training window
- `horizon`: forecast horizon in months

1. Put the function in a module under `src/models/` (one file per model family).
2. Register it in `MODEL_REGISTRY` in `src/models/__init__.py`.
3. Add its name to `ACTIVE_MODELS` in the notebook.

Errors inside a model are caught per run and recorded in the `error` column, so one bad
model doesn't stop the loop.

## Adding a feature track

Add an entry to `TRACK_CONFIG` in the notebook: `TRACK_CONFIG[track][target] = {"type", "path"}`.

- `"dataframe"`: a parquet of features indexed by month, sliced per fold. Use this for
  anything that can be precomputed without look-ahead (curated panels, reservoir embeddings).
- `"pipeline"`: a pickled, unfitted sklearn pipeline, fit on each fold's training slice of
  `stationary_panel.parquet`. Use this for anything that learns from the data (LASSO, PCA).

| Track | Type | Contents |
|---|---|---|
| A | dataframe | full 123-variable stationary panel |
| B | dataframe | 10–12 curated predictors per target (`scripts/build_track_b.py`) |
| C | pipeline | LASSO selection (tabled) |
| D / D_mini | pipeline | PCA, ~30 / 15 factors |
| QRC / CRC | dataframe | quantum / classical reservoir states over a trailing 12-month window of Track B |

Leakage rule: a precomputed track may only use information available at each row's
date. `build_quantum_tracks.py` scales inputs with expanding min/max for this reason.

## Evaluation protocol

- Fixed 120-month sliding training window, 1-month test window, 1-month step (647 folds)
- Metrics: MSE, MAE, median AE, RMSE, RMSPE, and MASE scaled by the in-sample MAE of the
  h-step naive forecast on the training window
- Results are also split by NBER recession (USREC) at the test date

## Team

| Member | Owns |
|---|---|
| Jack Morgan (`jhmorgan`) | Proposal · Literature review |
| David Giger (`digger`) | Bank & quantum context |
| Manuel Arce Aguirre (`arceaguirrem`) | Data pipeline · Classical baseline |
| Zihang "William" Niu (`william03niu`) | Feature selection & engineering |

Sponsor contacts (BMO Quantum Technologies): Richard Padbury · Guzmán Calleja ·
Rafael Martin-Cuevas.

Project documents (proposal, lit review, meeting notes, slides) are on `main` and in git history.

## References

- McCracken, M. & Ng, S. (2016). *FRED-MD: A Monthly Database for Macroeconomic Research.*
  Journal of Business & Economic Statistics.
- Konduri, T. & Li, Q. (2024). *Forecasting macroeconomic variables: A systematic comparison
  of machine learning methods.* University of Notre Dame JMP.
- Chen, S. Y.-C. et al. (2022). *Reservoir computing via quantum recurrent neural networks.*
  arXiv:2211.02612.
- Rigetti Computing (2023). *Recession prediction via signature kernels enhanced with quantum features.*
