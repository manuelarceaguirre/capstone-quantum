# Next Step Plan

## Where The Repo Stands

This repository is already strong on **data substrate** and **project framing**:

- `notebooks/01_data_prep.ipynb` builds a leakage-aware shared dataset.
- `data/processed/` contains the fold definitions, targets, and track artifacts needed for modeling.
- Sponsor docs and presentation notes clearly define the business question.

What is still missing is the **model-comparison layer**:

- There is no `02_classical_baseline.ipynb` yet.
- There is no reproducible script or notebook for benchmark evaluation across folds.
- There is no QRC/QSK execution entrypoint yet.

That means the highest-value next move is not more data prep. It is to establish a
stable **classical benchmark pipeline** that every later quantum result can be
compared against.

## What To Do Next

1. Lock down the classical benchmark first.
2. Use Track B as the first comparison lane for business-facing results.
3. Use D-mini as the first QRC-facing lane once the benchmark runner is stable.
4. Only after that, wire QSK and QRC notebooks/scripts onto the shared folds.

## Why This Is The Right Next Step

- BMO ultimately needs a recommendation on **whether quantum methods add value**.
- That claim is impossible to defend without a reproducible classical reference line.
- The repo already encodes the right folds and targets, so the bottleneck has moved
  from data engineering to evaluation engineering.

## What Was Executed

- Installed `pyarrow` locally so the processed parquet artifacts can be read.
- Added `scripts/run_classical_benchmark.py` as a reproducible benchmark runner.
- Confirmed a first pooled Track B starter baseline on horizon `h=1`:
  - `INDPRO` RMSE: `0.011108`
  - `PAYEMS` RMSE: `0.007466`
  - `CPIAUCSL` RMSE: `0.003095`
  - `S&P 500` RMSE: `0.036832`
  - `USREC` ROC-AUC: `0.872033`

These are not the final capstone numbers, but they are enough to prove the
evaluation path is viable and that Track B is already a strong candidate for an
early benchmark lane.

## Recommended Division Of Work

- You / baseline owner:
  Expand `scripts/run_classical_benchmark.py` into the official baseline runner and
  persist full result tables by track, target, and horizon.
- QRC owner:
  Start from `D-mini` only, using the same folds and pooled scoring logic.
- QSK owner:
  Start from `Track B` and `qsk_paths_track_B.npz`, again reusing the same fold
  logic and pooled classification scoring.

## Risks To Resolve Soon

- `Track B` still lacks sponsor-requested `PMI` and `LEI_chg`; currently you only
  have proxies/omissions documented in the notebook.
- Some housing/permit series still fail ADF in `metadata.json`.
- The saved sklearn pickles are version-sensitive, so recreating pipelines in code is
  safer than relying on unpickling across machines.
