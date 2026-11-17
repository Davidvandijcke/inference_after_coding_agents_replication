#!/usr/bin/env python3
"""Score the selected 12-month training contrast.

Fits OLS of training_m12 on treatment and site indicators, then reports the
treatment coefficient with an HC2 heteroskedasticity-robust standard error and
two-sided t-test p-value.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"training_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    fitted = smf.ols("training_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
