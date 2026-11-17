#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on partner employment plans."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "partner_employment_1y_fu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {OUTCOME, "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    analysis = data[[OUTCOME, "treatment", "block"]].dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(block)", data=analysis).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
