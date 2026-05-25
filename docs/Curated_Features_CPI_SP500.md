# Curated Feature Set for CPI Inflation and S&P 500 Forecasting

## Recommendation

For the next round of RNN/GRU and reservoir-style forecasting experiments, I recommend using one unified curated feature set with 10 variables. This keeps the input space small enough for limited-data modeling while still covering the main macro-financial channels relevant to both CPI inflation and the S&P 500.

| Feature | Primary channel | Why it is included |
|---|---|---|
| `CPIAUCSL` | Inflation persistence | Headline CPI is the direct target family for inflation forecasting, and lagged inflation is usually one of the strongest baseline signals. |
| `PCEPI` | Alternative price pressure | PCE inflation provides a broader consumption-price cross-check and is closely tied to the Fed's inflation-monitoring framework. |
| `PPICMM` | Upstream cost pressure | Producer-price commodities can capture cost-push pressure before it passes through to consumer prices. |
| `OILPRICEx` | Energy shock | Oil prices affect headline inflation directly and can also affect equity valuations through costs, margins, and discount-rate expectations. |
| `FEDFUNDS` | Policy stance | The policy rate summarizes monetary tightening/easing, which affects inflation, demand, discount rates, and equity prices. |
| `T10YFFM` | Yield-curve / expectations | The 10-year Treasury minus fed funds spread captures term-structure expectations and the market's view of policy tightness. |
| `UNRATE` | Labor-market slack | Unemployment is a classic slack variable connected to inflation pressure and recession risk. |
| `PERMIT` | Forward-looking real activity | Building permits are interest-rate sensitive and forward-looking, making them useful for business-cycle and demand conditions. |
| `S&P PE ratio` | Equity valuation | The price-to-earnings ratio is a compact valuation signal for expected equity returns and discount-rate pressure. |
| `VIXCLSx` | Financial uncertainty | VIX captures forward-looking equity-market risk and stress, which is especially relevant for S&P 500 forecasting. |

## Selection Logic

I used a hybrid selection approach rather than a purely statistical or purely theoretical approach. The reason is that our project needs a feature set that is both empirically defensible and easy to justify in a presentation.

1. Economic channel coverage: I first mapped candidate variables into interpretable channels: inflation persistence, cost-push pressure, energy shocks, monetary policy, term structure, labor slack, housing/activity, equity valuation, and financial uncertainty.

2. Statistical screening: I screened all 123 stationarized predictors in the processed FRED-MD-style panel against CPI and S&P 500 targets at horizons 1, 3, 6, and 12 months. The script computes absolute correlation, mutual information, and expanding-window LASSO selection frequency.

3. Stability across targets and horizons: I avoided choosing features that only look good for one narrow horizon unless they have a strong economic reason. The final set is intended to be robust for both target families.

4. Redundancy control: Several variables scored highly statistically, especially interest-rate spreads, housing subcomponents, inventories, and sector-specific activity measures. I did not include all of them because they overlap heavily with selected representatives like `T10YFFM`, `PERMIT`, and `UNRATE`.

## Alternative Approaches We Could Defend

1. Pure theory-driven set: Choose variables only from macro-financial logic. This is easiest to explain but may ignore dataset-specific predictive patterns.

2. Pure statistical filter: Rank variables by correlation or mutual information with the target. This is transparent and fast, but it can select redundant features and may overfit noisy relationships.

3. Embedded model selection: Use LASSO or elastic net in each expanding training window and keep variables selected consistently. This is more prediction-oriented, but selected variables can be harder to explain economically.

4. Hybrid curated approach: Combine economic channel coverage with statistical screening. This is the recommended approach because our capstone needs both forecasting usefulness and a clear justification for BMO-facing discussion.

5. Target-specific feature sets: Build one set for inflation and a separate set for S&P 500. This may improve performance, but it complicates comparison across RNN/GRU/reservoir models. I recommend starting with the unified set, then testing target-specific variants later.

## Reproducible Outputs

The screening script is:

```bash
~/.virtualenvs/CV_Work/bin/python scripts/select_curated_features_cpi_sp500.py
```

Generated files:

| File | Purpose |
|---|---|
| `results/curated_features_cpi_sp500/recommended_features.csv` | Final 10-feature recommendation with scores and explanations. |
| `results/curated_features_cpi_sp500/feature_screening_summary.csv` | One-row-per-feature summary across both targets and all horizons. |
| `results/curated_features_cpi_sp500/feature_screening_detail.csv` | Detailed target-horizon-level screening results. |

## Key Takeaway

The final 10-feature set is not just a performance screen. It is a compact macro-financial representation of the economy: prices, costs, energy, policy, rates, labor slack, housing demand, equity valuation, and market risk. That makes it suitable for limited-data sequence models where we want useful nonlinear transformations without feeding the model a very large, noisy panel.

## References for Justification

- McCracken and Ng's FRED-MD paper motivates using a broad monthly macroeconomic panel as the starting point for empirical macro forecasting: https://files.stlouisfed.org/files/htdocs/fred-databases/fredmd.pdf
- Federal Reserve discussions of Phillips-curve logic support including labor-market slack variables such as unemployment for inflation forecasting: https://www.federalreserve.gov/econres/notes/feds-notes/nonlinear-phillips-curves-20240904.html
- Campbell and Shiller's valuation-ratio work supports including valuation ratios such as P/E-style measures for equity-return forecasting: https://www.nber.org/papers/w8221
