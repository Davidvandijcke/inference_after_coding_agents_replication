#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment effect.

Usage: python score.py --data <path> --output <path>

Fits employment at month 12 on treatment and site indicators, then reports the
treatment coefficient, HC2 robust standard error, and two-sided t-test p-value.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))

    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
