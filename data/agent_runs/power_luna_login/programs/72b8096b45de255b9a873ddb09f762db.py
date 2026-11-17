#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    # OLS with site indicators; HC2 robust standard errors, as specified in
    # CODEBOOK.md. The treatment test is a two-sided t-test.
    site_indicators = pd.get_dummies(
        data["site"].astype("category"), prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data["treatment"].astype(float).rename("treatment"),
            site_indicators,
        ],
        axis=1,
    )
    model = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=model.df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
