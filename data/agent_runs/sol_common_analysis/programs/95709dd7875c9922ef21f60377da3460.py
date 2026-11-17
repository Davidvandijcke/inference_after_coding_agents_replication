#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on 12-month food insecurity.

Requires pandas and statsmodels (and their dependencies).
OLS includes an intercept, treatment, and site indicators. Inference uses
HC2 standard errors and a two-sided t-test with residual degrees of freedom.
The reported p-value is nominal; this program runs only the selected test.
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
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")

    data = pd.read_csv(
        args.data, usecols=["site", "treatment", "food_insecure_m12"]
    )
    if data.empty or data.isna().any().any():
        parser.error("The analysis requires nonempty, complete observations")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1")
    if not data["food_insecure_m12"].isin([0, 1]).all():
        parser.error("food_insecure_m12 must be binary (0 or 1)")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design.insert(0, "intercept", 1.0)
    fit = sm.OLS(data["food_insecure_m12"], design).fit(
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
