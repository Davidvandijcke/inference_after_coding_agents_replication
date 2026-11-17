#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment treatment effect."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    result = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
