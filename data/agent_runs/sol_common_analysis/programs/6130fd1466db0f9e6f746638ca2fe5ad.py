"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. The estimate and standard error are in
probability units. The p-value is the unadjusted, two-sided HC2 t-test p-value.
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

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The required analysis columns must be complete and nonempty.")
    if not set(data["treatment"].unique()) == {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employment at 12 months must be coded 0 or 1.")

    result = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
