#!/usr/bin/env python3
"""Score the prespecified 12-month employment contrast in a Pathways dataset."""

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
    required = {"treatment", "site", "employed_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    # Site indicators reflect the within-site randomization. HC2 robust
    # standard errors and a two-sided t-test follow CODEBOOK.md.
    result = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2"
    )
    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
