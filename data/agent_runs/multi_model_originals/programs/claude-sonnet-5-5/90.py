"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test: effect of treatment on training_m6 (enrolled in education or job
training in the survey week, 6 months after random assignment), full sample.
Convention per CODEBOOK.md: OLS of the outcome on treatment plus site
indicators, HC2 robust standard errors, two-sided t-test on the treatment
coefficient. No covariates, no subgroup or outcome search.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "training_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    res = smf.ols(f"{OUTCOME} ~ treatment + C(site)", df).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(res.params["treatment"]),
        "std_error": float(res.bse["treatment"]),
        "p_value": float(res.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
