#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed in advance; no search is repeated here):
    Full sample, outcome = training_m12 (enrolled in education/job training at
    the 12-month follow-up). OLS of training_m12 on treatment plus site
    indicators, HC2 robust standard errors, two-sided t-test of the treatment
    coefficient (df = n - number of regressors).

Usage:
    python score.py --data <path to csv> --output <path to json>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "training_m12"


def run(data_path):
    d = pd.read_csv(data_path)
    d = d[[OUTCOME, "treatment", "site"]].dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=d).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = run(args.data)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
