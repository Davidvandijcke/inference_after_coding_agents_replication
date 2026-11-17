#!/usr/bin/env python3
"""Score the year 1 program effect on physician/clinic/ER visits.

Uses the codebook's block-adjusted, baseline-adjusted OLS specification and
HC2 heteroskedasticity-robust standard errors. Missing observations are omitted.
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
    required = {
        "physician_or_er_visit_y1",
        "physician_or_er_visit_base",
        "treatment",
        "block",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    analysis = data[list(required)].dropna()
    fit = smf.ols(
        "physician_or_er_visit_y1 ~ treatment + C(block) + physician_or_er_visit_base",
        data=analysis,
    ).fit(cov_type="HC2")
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
