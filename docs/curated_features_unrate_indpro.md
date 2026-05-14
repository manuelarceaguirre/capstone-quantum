# Curated predictor sets for UNRATE and INDPRO

Prepared for the Week 7 / Thursday task: **find defensible sources for a small curated set of predictors for unemployment and industrial production**.

## Executive summary

We should use two small, source-justified feature sets rather than treating Track B as arbitrary. The guiding principle is:

- keep each set to **6-10 monthly predictors**;
- prefer variables that are either direct target persistence signals, Conference Board leading indicators, Federal Reserve industrial-production source concepts, or standard macro-forecasting variables;
- make every transformation transparent: level features come from `levels_panel.parquet`, while stationary features come from `stationary_panel.parquet`.

The machine-readable version is in `configs/curated_feature_sets.json`; helper code is in `src/classical_utils/curated_features.py`.

---

## Sources used

| Source | What it supports |
|---|---|
| The Conference Board, **Leading Economic Index (LEI) for the US**: https://www.conference-board.org/topics/us-leading-indicators/ | LEI is intended to anticipate business-cycle turning points. Its ten components include manufacturing hours, initial claims, manufacturers' new orders, building permits, S&P 500, interest-rate spread, and consumer expectations. |
| Federal Reserve, **Industrial Production and Capacity Utilization Methodology**: https://www.federalreserve.gov/releases/g17/Meth/MethIP.htm | IP covers manufacturing, mining, and utilities. Monthly IP uses physical product data where possible and production-input measures such as production-worker hours when direct product data are unavailable. |
| Federal Reserve, **Capacity Utilization notes**: https://www.federalreserve.gov/RELEASES/g17/CapNotes.htm | Capacity utilization is output relative to sustainable maximum output, making it directly tied to industrial production intensity. |
| Stock and Watson (2002), **Macroeconomic Forecasting Using Diffusion Indexes**: https://ideas.repec.org/a/bes/jnlbes/v20y2002i2p147-62.html | Supports forecasting macro variables from broad monthly macro panels / reduced predictor sets. |
| RBA Bulletin (2025), **How Useful are Leading Labour Market Indicators at Forecasting the Unemployment Rate?**: https://www.rba.gov.au/publications/bulletin/2025/apr/how-useful-are-leading-labour-market-indicators-at-forecasting-the-unemployment-rate.html | Supports using leading labor-market indicators for unemployment-rate forecasts. |

---

## UNRATE curated predictors

**Target:** civilian unemployment rate.

| Feature | Project source | Meaning | Why defensible |
|---|---|---|---|
| `UNRATE_level` | `levels_panel.UNRATE` | Current unemployment rate level | Unemployment is persistent; current level is the natural baseline state. |
| `UNRATE_delta` | `stationary_panel.UNRATE` | Monthly change in unemployment | Captures recent labor-market momentum. |
| `CLAIMSx` | `stationary_panel.CLAIMSx` | Initial unemployment claims | Initial claims are an LEI component and a direct labor-market stress indicator. |
| `HWIURATIO` | `stationary_panel.HWIURATIO` | Help-wanted / unemployed ratio | Proxies labor demand relative to labor supply. |
| `PAYEMS` | `stationary_panel.PAYEMS` | Nonfarm payroll employment growth | Payroll employment is central to labor-market conditions and the coincident business cycle. |
| `PERMIT` | `stationary_panel.PERMIT` | New housing permits | LEI component; housing weakness can precede weaker construction/labor demand. |
| `S&P 500` | `stationary_panel.S&P 500` | Equity-market return/growth signal | LEI component; forward-looking financial expectations. |
| `T10Y3M_level` | `levels_panel.GS10 - levels_panel.TB3MS` | 10-year minus 3-month Treasury spread | Term spread captures recession/financial-conditions signal. |
| `UMCSENTx` | `stationary_panel.UMCSENTx` | Consumer sentiment | Consumer expectations are an LEI component and can precede spending/hiring weakness. |

### UNRATE comment for presentation

This set combines four channels: labor-market persistence, direct labor-demand stress, cyclical housing/consumer demand, and financial-market expectations. It is intentionally interpretable and small.

---

## INDPRO curated predictors

**Target:** industrial production.

| Feature | Project source | Meaning | Why defensible |
|---|---|---|---|
| `INDPRO_level` | `levels_panel.INDPRO` | Current industrial production level | Captures production persistence and current industrial-cycle state. |
| `INDPRO_growth` | `stationary_panel.INDPRO` | Recent industrial production growth | Captures short-run production momentum. |
| `IPMANSICS` | `stationary_panel.IPMANSICS` | Manufacturing industrial production | Manufacturing is a major part of aggregate IP. |
| `CUMFNS` | `stationary_panel.CUMFNS` | Capacity utilization | Fed capacity-utilization measure is directly tied to output relative to sustainable capacity. |
| `AWHMAN` | `stationary_panel.AWHMAN` | Average weekly hours in manufacturing | LEI component; Fed IP methodology uses production-worker hours as input indicators where product data are unavailable. |
| `AMDMNOx` | `stationary_panel.AMDMNOx` | Manufacturers' new orders, durable goods | New orders are an LEI component and a forward-looking demand signal for production. |
| `CMRMTSPLx` | `stationary_panel.CMRMTSPLx` | Real manufacturing and trade sales | Demand-side coincident signal for goods production. |
| `PERMIT` | `stationary_panel.PERMIT` | Building permits | LEI component; construction and housing demand lead materials/durable-goods production. |
| `S&P 500` | `stationary_panel.S&P 500` | Equity-market return/growth signal | LEI component; forward-looking expectations for activity/investment. |
| `T10Y3M_level` | `levels_panel.GS10 - levels_panel.TB3MS` | 10-year minus 3-month Treasury spread | Financial-conditions / recession signal relevant for demand and production cycles. |

### INDPRO comment for presentation

This set follows the logic of the Fed IP methodology and the Conference Board LEI: production persistence, manufacturing activity, capacity utilization, production-worker hours, new orders, construction demand, and financial expectations.

---

## Assumptions and caveats

1. **Current-level target predictors are included deliberately.** `UNRATE_level` and `INDPRO_level` are current values known at forecast origin. They are important because level forecasts are persistent; they should not be confused with leaked future values.
2. **Stationary features remain transformed.** Most non-level columns come from `stationary_panel.parquet`, so names like `PAYEMS` or `INDPRO_growth` represent transformed growth/change concepts, not raw levels.
3. **T10Y3M is derived.** It is not a direct FRED-MD column. We construct it as `GS10 - TB3MS` from raw levels because yield-curve inversion is a level state.
4. **This is not an optimized feature-selection result.** The goal is defensibility for a small-data capstone setting. Full-panel PCA/LASSO can still be evaluated separately.
5. **If the team standardizes on fixed training windows, use these features inside that standard loop.** Feature justification is independent of the final fold convention.

---

## How to load these panels

```python
from src.classical_utils.curated_features import build_curated_panel

X_unrate = build_curated_panel("UNRATE")
X_indpro = build_curated_panel("INDPRO")
```

Expected shapes on the current processed data:

- `UNRATE`: 767 monthly rows × 9 features
- `INDPRO`: 767 monthly rows × 10 features
