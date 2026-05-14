# Thursday deliverables summary

## Week7 tasks addressed

- Wrap models as functions that take train/test dataframes and return predictions with dates.
- Prepare for David's shared evaluation notebook by standardizing prediction output.
- Add metrics helpers from the Week7 metric list, including MASE/RMSSE.
- Source and document curated predictors for unemployment and industrial production.

## Main files added

### Curated features

- `docs/curated_features_unrate_indpro.md`  
  Short source-backed explanation for curated `UNRATE` and `INDPRO` predictors.

- `configs/curated_feature_sets.json`  
  Machine-readable feature list with source columns, reasons, and source URLs.

- `src/classical_utils/curated_features.py`  
  Helper to build curated feature panels from processed data.

- `data/processed/track_B_unrate_curated.parquet`  
  UNRATE curated panel: 767 rows x 9 features.

- `data/processed/track_B_indpro_curated.parquet`  
  INDPRO curated panel: 767 rows x 10 features.

### Model wrappers and metrics

- `src/classical_utils/model_wrappers.py`  
  Standard model wrapper functions. Output columns: `date`, `target`, `horizon`, `model`, `y_true`, `y_pred`.

- `src/classical_utils/metrics.py`  
  Implements MSE, MAE, MAPE, sMAPE, MSLE, median absolute error, MASE, and RMSSE.

- `docs/thursday_model_interface.md`  
  Notes on the shared train/test dataframe interface and MASE convention.

## Curated feature sets

### UNRATE

Features:

- `UNRATE_level`
- `UNRATE_delta`
- `CLAIMSx`
- `HWIURATIO`
- `PAYEMS`
- `PERMIT`
- `S&P 500`
- `T10Y3M_level`
- `UMCSENTx`

Reason: combines unemployment persistence, direct labor-market stress, labor demand, housing cycle, financial expectations, yield curve, and consumer expectations.

### INDPRO

Features:

- `INDPRO_level`
- `INDPRO_growth`
- `IPMANSICS`
- `CUMFNS`
- `AWHMAN`
- `AMDMNOx`
- `CMRMTSPLx`
- `PERMIT`
- `S&P 500`
- `T10Y3M_level`

Reason: combines production persistence, manufacturing activity, capacity utilization, production-worker hours, new orders, demand, housing, and financial conditions.

## Sources used

- Conference Board Leading Economic Index components
- Federal Reserve Industrial Production methodology
- Federal Reserve Capacity Utilization notes
- Stock and Watson diffusion-index macro forecasting
- RBA labor-market leading indicators note

## Important caveat

The team still needs to agree on the fixed-window convention:

- fixed input/history window, or
- fixed training window.

For MASE/RMSSE, the recommended convention is to compute the naive scale fold-by-fold from the training target, then pool scaled test errors.
