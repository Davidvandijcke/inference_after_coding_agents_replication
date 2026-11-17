#!/usr/bin/env python3
"""Estimate the 12-month earnings ITT, adjusted for site and baseline earnings.

Requires pandas and statsmodels. Runs only this fixed analysis.
"""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["earnings_m12", "treatment", "site", "earnings_baseline"]
    if data[required].isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment", "earnings_baseline"]], sites], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    fit = sm.OLS(data["earnings_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
