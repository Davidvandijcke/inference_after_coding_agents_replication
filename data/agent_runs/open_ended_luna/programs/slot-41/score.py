#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Fits hours_m6 on treatment and site indicators among participants with some
college (education == 3), using HC2 standard errors and a two-sided t-test.
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
    required = {"education", "hours_m6", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    subgroup = data.loc[data["education"] == 3]
    if subgroup.empty:
        raise ValueError("No participants with education == 3 (some college)")

    result = smf.ols(
        "hours_m6 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
