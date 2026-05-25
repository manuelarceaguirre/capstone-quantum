# Thursday modeling interface plan

This note translates the Week 7 tasks into a concrete interface that David's standard evaluation notebook can call.

## Week 7 tasks from `Week7.md`

1. **Everyone:** wrap each model in a function that imports a train + test dataframe, then outputs predictions and dates.
2. **David:** build one standard notebook that loops through tracks, target variables, horizons, and POOS train/test splits, then saves metrics in a standardized format.
3. **Jack:** compile performance metrics and curated features for each target variable.
4. **Manuel:** justify curated predictors for unemployment and industrial production.
5. **William:** justify curated predictors for CPI/inflation.

## Deliverables in this branch

| Deliverable | Path | Purpose |
|---|---|---|
| Curated feature source note | `docs/curated_features_unrate_indpro.md` | Human-readable justification for UNRATE and INDPRO predictors. |
| Machine-readable feature config | `configs/curated_feature_sets.json` | Feature lists, transformations, source columns, and source URLs. |
| Curated panel loader | `src/classical_utils/curated_features.py` | Builds `UNRATE` and `INDPRO` panels from processed artifacts. |
| Model wrappers | `src/classical_utils/model_wrappers.py` | Standard functions that accept train/test dataframes and return prediction rows. |
| Metrics helper | `src/classical_utils/metrics.py` | Implements Week7 metric list, including MASE/RMSSE scale conventions. |

## Standard prediction output schema

Every model wrapper should return a dataframe with exactly these core columns:

| Column | Meaning |
|---|---|
| `date` | Forecast target date / test observation date. |
| `target` | Target column name. |
| `horizon` | Forecast horizon in months. |
| `model` | Model name. |
| `y_true` | Observed target value. |
| `y_pred` | Model prediction. |

Additional columns are allowed, but these six should be present for David's evaluator.

## Wrapper signature

```python
def model_fn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    **kwargs,
) -> pd.DataFrame:
    ...
```

The train/test dataframes should already contain horizon-aligned target columns. The wrapper should not know about global folds or notebooks.

## Implemented wrappers

`src.classical_utils.model_wrappers` includes:

- `predict_naive_last_value`
- `predict_linear_regression`
- `predict_sklearn_model`
- `supported_model_wrappers`

Supported model names:

- `LinearRegression`
- `AdaBoost`
- `Random Forest`
- `AdaBoost-DI`
- `kNN-DI`
- `SVR linear-DI`
- `XGBoost-DI` if `xgboost` is installed

## MASE / RMSSE convention

To align with William's workstream, the recommended convention is:

1. For each fold, compute the naive scale from the **training target only**:
   - MASE denominator: `mean(abs(y_train[t] - y_train[t-1]))`
   - RMSSE denominator: `mean((y_train[t] - y_train[t-1])**2)`
2. Compute scaled test errors using that fold-specific denominator.
3. Pool scaled errors across test predictions.

This is implemented in `src.classical_utils.metrics.mase_scale`, `rmsse_scale`, and `regression_metrics`.

## Known evaluation question still requiring team agreement

The team still needs one explicit convention on fixed windows:

- **Fixed input window:** every sample uses the same trailing history length, but model training can use expanding or rolling folds.
- **Fixed training window:** each fold truncates the training sample to a fixed number of months, as William's RNN/GRU script does.

For model comparability, David's standard notebook should choose one convention and record it in every output manifest.
