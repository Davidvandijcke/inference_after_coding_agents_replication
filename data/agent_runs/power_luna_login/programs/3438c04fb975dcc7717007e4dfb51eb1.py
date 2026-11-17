#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "site", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))
    if data[sorted(required)].isna().any().any():
        parser.error("required analysis columns must not contain missing values")

    # Site indicators reflect the within-site randomization. HC2 gives the
    # heteroskedasticity-robust standard error specified in the codebook.
    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(cov_type="HC2")
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=float(fit.df_resid)))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
