from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Track B target-specific files for Jack's multi-file evaluator.
# Columns are selected from track_A_full.parquet so names and transformations
# match Track A exactly; only the number of predictors is curated/smaller.
FEATURE_SETS = {
    "INDPRO": [
        "INDPRO",
        "IPMANSICS",
        "CUMFNS",
        "AWHMAN",
        "AMDMNOx",
        "CMRMTSPLx",
        "PERMIT",
        "S&P 500",
        "GS10",
        "TB3MS",
    ],
    "PAYEMS": [
        "PAYEMS",
        "UNRATE",
        "CLAIMSx",
        "HWIURATIO",
        "PERMIT",
        "S&P 500",
        "GS10",
        "TB3MS",
        "UMCSENTx",
        "CES0600000007",
    ],
    "CPIAUCSL": [
        "CPIAUCSL",
        "CPIULFSL",
        "CPIAPPSL",
        "CPITRNSL",
        "CPIMEDSL",
        "OILPRICEx",
        "PPICMM",
        "PCEPI",
        "FEDFUNDS",
        "M2SL",
        "UNRATE",
        "S&P 500",
    ],
    "S&P 500": [
        "S&P 500",
        "VIXCLSx",
        "FEDFUNDS",
        "TB3MS",
        "GS10",
        "AAA",
        "BAA",
        "COMPAPFFx",
        "UMCSENTx",
        "INDPRO",
        "PAYEMS",
        "CPIAUCSL",
    ],
}


def main() -> None:
    track_a = pd.read_parquet(DATA_DIR / "track_A_full.parquet").sort_index()
    for target, columns in FEATURE_SETS.items():
        missing = [column for column in columns if column not in track_a.columns]
        if missing:
            raise KeyError(f"{target} feature set has missing Track-A columns: {missing}")
        out = track_a[columns].copy()
        path = DATA_DIR / f"track_b_{target}.parquet"
        out.to_parquet(path)
        print(f"wrote {path.relative_to(ROOT)} shape={out.shape}")


if __name__ == "__main__":
    main()
