#!/usr/bin/env python
"""Pathways to Work: rerun the chosen analysis on a dataset with the same columns.

Chosen test (fixed in advance, no search):
  Outcome:   employed_m12 (employment in the survey week, 12 months after random assignment)
  Sample:    all participants
  Model:     OLS of employed_m12 on treatment + site indicators
  Inference: HC2 robust standard error, two-sided t-test of the treatment coefficient
             (residual degrees of freedom)

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    missing = {OUTCOME, "treatment", "site"} - set(d.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    d = d[[OUTCOME, "treatment", "site"]].dropna()

    formula = f"{OUTCOME} ~ treatment + C(site)" if d["site"].nunique() > 1 else f"{OUTCOME} ~ treatment"
    fit = smf.ols(formula, data=d).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
