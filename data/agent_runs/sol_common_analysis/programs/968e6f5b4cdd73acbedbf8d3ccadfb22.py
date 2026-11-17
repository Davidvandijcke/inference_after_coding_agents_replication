"""Estimate the full-sample effect on 12-month employment only.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels (and their dependencies).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["treatment", "site", "employed_m12"]
    if any(column not in data.columns for column in required):
        raise ValueError(f"Data must contain {required}")
    if data[required].isna().any().any():
        raise ValueError("Analysis variables must be fully observed")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
