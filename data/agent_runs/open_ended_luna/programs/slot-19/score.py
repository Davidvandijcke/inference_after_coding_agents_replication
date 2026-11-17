#!/usr/bin/env python3
"""Score the prespecified lead analysis for Pathways to Work.

Estimates the treatment effect on food insecurity at month 12 using OLS with
site indicators and HC2 heteroskedasticity-robust standard errors.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


OUTCOME = "food_insecure_m12"


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", OUTCOME}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Analysis columns contain missing values")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1")
    if not set(data[OUTCOME].unique()).issubset({0, 1}):
        raise ValueError(f"{OUTCOME} must be binary (0/1)")

    site_indicators = pd.get_dummies(
        data["site"].astype("category"),
        prefix="site",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [data[["treatment"]].astype(float), site_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    fitted = sm.OLS(data[OUTCOME].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )

    return {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
