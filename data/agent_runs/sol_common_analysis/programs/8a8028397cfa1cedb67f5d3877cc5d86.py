"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. The fixed specification is OLS of
employed_m12 on treatment and site indicators, with HC2 standard errors
and a two-sided Student t test using residual degrees of freedom.
The reported p-value is for this single test, without multiplicity adjustment.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data)
    required = ["employed_m12", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        parser.error(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        parser.error("The analysis columns must be fully observed.")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1, and no other values.")

    fit = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    if fit.df_resid <= 0:
        parser.error("Insufficient residual degrees of freedom.")
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
