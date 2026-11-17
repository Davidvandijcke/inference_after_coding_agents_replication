#!/usr/bin/env python3
"""Score the 12-month employment treatment effect using the packet convention."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # OLS of the outcome on treatment and site indicators; HC2 robust inference.
    site_indicators = pd.get_dummies(
        data["site"].astype("category"), drop_first=True, dtype=float
    )
    design = pd.concat(
        [data[["treatment"]].astype(float), site_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    result = sm.OLS(data["employed_m12"].astype(float), design).fit(cov_type="HC2")

    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted program effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
