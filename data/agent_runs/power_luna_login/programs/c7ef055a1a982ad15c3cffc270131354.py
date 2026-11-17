#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV input with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = {"employed_m12", "treatment", "site"}
    missing = sorted(needed.difference(data.columns))
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))
    if data[list(needed)].isna().any().any():
        parser.error("required columns must not contain missing values")

    result = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(cov_type="HC2")
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
