"""Pathways to Work: rerun the chosen analysis on a dataset with the same columns.

Chosen test (fixed; no search is performed here):
    Effect of assignment to the program on debt_collections_m12 (1 if the
    participant had a debt in collections in the past 6 months, measured at
    the 12-month follow-up), full sample.
    OLS: debt_collections_m12 ~ treatment + site indicators (no other covariates).
    HC2 heteroskedasticity-robust standard error; two-sided t-test of the
    treatment coefficient (residual df = n - k).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m12"
FORMULA = f"{OUTCOME} ~ treatment + C(site)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    missing = [c for c in (OUTCOME, "treatment", "site") if c not in d.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    d = d[[OUTCOME, "treatment", "site"]].dropna()

    fit = smf.ols(FORMULA, data=d).fit(cov_type="HC2", use_t=True)
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
